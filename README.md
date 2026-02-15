# Naive Retrieval-Augmented Generation (RAG) System

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline that answers salary-related queries using semantic retrieval and large language model grounding.

The system retrieves relevant rows from a structured dataset using FAISS vector similarity search and injects the retrieved context into a Gemini LLM prompt to generate grounded responses.

---

## Overview

Pipeline:

1. Dataset preprocessing (column filtering and null removal)
2. Embedding generation using SentenceTransformers (all-MiniLM-L6-v2)
3. FAISS vector index creation
4. Top-k semantic retrieval
5. Context injection into Gemini prompt
6. Grounded response generation

---

## Technologies Used

- Python
- Pandas
- NumPy
- SentenceTransformers
- FAISS
- Google Gemini (LLM)
- Prompt Engineering

---

## Example Queries

- What is the average salary of data scientists with over 5 years of experience?
- How does experience level impact salary?
- What are the top locations for data scientist jobs?

---

## Setup

Install dependencies:

pip install -r requirements.txt

Set Gemini API key:

Windows:
setx API_KEY "your_api_key_here"

Mac/Linux:
export API_KEY="your_api_key_here"

Run:

python naive_rag.py

---

## What This Demonstrates

- End-to-end RAG pipeline implementation
- Embedding-based semantic retrieval
- Vector indexing with FAISS
- Context-grounded LLM responses
- Structured data preprocessing and filtering
