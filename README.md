# Production-Grade RAG Agent (FastAPI + Hybrid Retrieval)

A production-ready Retrieval-Augmented Generation (RAG) system built using FastAPI, hybrid retrieval (FAISS + BM25), cross-encoder re-ranking, and open-source LLMs.

This system is designed to provide accurate, context-aware responses over custom document corpora while minimizing hallucinations through layered retrieval and evaluation mechanisms.

---

## 🚀 Features

- Hybrid Retrieval (Vector + Keyword Search)
- SentenceTransformer Embeddings (all-MiniLM-L6-v2)
- FAISS Vector Indexing
- BM25 Keyword Ranking
- Cross-Encoder Re-ranking
- Hallucination Detection Pipeline
- LLM-Based Response Evaluation
- Multi-Agent Business Analysis Layer
- Dockerized Deployment
- Hugging Face Spaces Deployment Ready

---

## 🏗️ System Architecture

User Query  
→ Hybrid Retrieval (FAISS + BM25)  
→ Cross-Encoder Re-ranking  
→ Context Construction  
→ LLM Generation (Mistral / TinyLlama / FLAN-T5)  
→ Hallucination Detection  
→ Final Response  

---

## 🧠 Tech Stack

### Retrieval & Embeddings
- SentenceTransformers (all-MiniLM-L6-v2)
- FAISS
- BM25

### Re-ranking
- Cross-Encoder (SentenceTransformers)

### LLMs
- Mistral
- TinyLlama
- FLAN-T5

### Backend
- Python
- FastAPI

### Deployment
- Docker
- Hugging Face Spaces

---

## 📂 Project Structure
├── main.py
├── rag_pipeline.py
├── retriever.py
├── reranker.py
├── hallucination_checker.py
├── requirements.txt
├── Dockerfile
└── data/
