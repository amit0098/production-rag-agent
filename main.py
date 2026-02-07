from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
import numpy as np
import torch
import faiss

# ------------------------
# Load Models
# ------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI(title="Production-grade RAG Agent")

chunks = []
faiss_index = None
bm25 = None

# ------------------------
# Helper Functions
# ------------------------
def llm_call(prompt: str) -> str:
    output = llm_pipeline(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()


def ingest_pdf(path):
    global chunks, faiss_index, bm25

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    bm25 = BM25Okapi([t.split() for t in texts])


def hybrid_search(query, k=5):
    query_emb = embedding_model.encode([query])
    _, vec_ids = faiss_index.search(query_emb, k)

    bm25_scores = bm25.get_scores(query.split())
    bm25_ids = np.argsort(bm25_scores)[-k:]

    ids = list(set(vec_ids[0]) | set(bm25_ids))
    return [chunks[i].page_content for i in ids]


def hallucination_check(answer, context):
    prompt = f"""
Check if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Reply YES or NO.
"""
    return "YES" in llm_call(prompt).upper()


def evaluate(answer, context):
    prompt = f"""
Rate the answer on relevance, faithfulness, and clarity (0-10).

Context:
{context}

Answer:
{answer}
"""
    return llm_call(prompt)


def business_agents(answer):
    roles = {
        "Market Analyst": "Analyze market implications",
        "Risk Analyst": "Identify risks",
        "Finance Analyst": "Analyze financial impact",
        "Strategy Agent": "Give final recommendation"
    }

    results = {}
    for role, task in roles.items():
        prompt = f"You are a {role}. {task}.\n\nInput:\n{answer}"
        results[role] = llm_call(prompt)

    return results

# ------------------------
# API Schema
# ------------------------
class QueryRequest(BaseModel):
    question: str

# ------------------------
# API Endpoints
# ------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    ingest_pdf(path)
    return {"status": "Document ingested"}


@app.post("/query")
def query(req: QueryRequest):
    docs = hybrid_search(req.question)
    context = "\n\n".join(docs)

    prompt = f"""
You are a business analyst.

Use ONLY the context below.
If unsure, say "I don't have enough information."

Context:
{context}

Question:
{req.question}

Answer:
"""
    answer = llm_call(prompt)

    return {
        "answer": answer,
        "hallucination_free": hallucination_check(answer, context),
        "evaluation": evaluate(answer, context),
        "multi_agent_analysis": business_agents(answer)
    }
