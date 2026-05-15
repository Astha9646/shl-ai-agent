# SHL Conversational Assessment Recommendation System

## Overview

This project is a conversational AI system that recommends SHL assessments based on hiring requirements provided in natural language.

The system supports:
- conversational recommendations
- clarification handling
- assessment comparison
- grounded responses
- hybrid retrieval
- stateless API interaction

---

## Architecture

User Query
→ FastAPI Backend
→ Guardrails
→ Conversation State Extraction
→ Hybrid Retrieval (BM25 + FAISS)
→ LLM Response Generation
→ Structured JSON Response

---

## Retrieval System

The retrieval pipeline combines:

1. BM25 keyword retrieval
2. FAISS semantic vector search

Weighted hybrid ranking is used to improve relevance and reduce noisy recommendations.

Embeddings are generated using:
- sentence-transformers/all-MiniLM-L6-v2

---

## LLM Layer

The conversational layer uses:
- OpenRouter API
- GPT-4o-mini

The LLM is grounded strictly on retrieved SHL catalog entries to reduce hallucinations.

---

## Guardrails

The system includes:
- prompt injection detection
- off-topic request detection
- legal advice filtering
- vague query clarification handling

---

## API Endpoints

### GET /health
Health check endpoint.

### POST /chat
Conversational recommendation endpoint.

---

## Evaluation

The system was tested for:
- vague queries
- technical role recommendations
- conversational refinement
- comparison queries
- prompt injection attempts

---

## Deployment

The API is deployed using:
- FastAPI
- Uvicorn
- Render