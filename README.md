# Finanical Rag

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Dagster for orchestration. The pipeline ingests a folder of PDFs, processes them into retrievable chunks, embeds them into vector representations, and stores them in both a document store (SQLite) and a vector store (ChromaDB).

## Features

- Automated Ingestion: Monitors a folder of PDFs and ingests new documents.
- Chunking: Splits large documents into smaller, semantically meaningful text chunks.
- Embeddings: Generates vector embeddings using transformer-based models.
- Dual Storage:
    - SQLite Document Store → for metadata and raw chunks.
    - ChromaDB Vector Store → for similarity search and retrieval.
- Orchestration with Dagster: Modular, production-ready pipeline with observability and reproducibility.

## Tech Stack

- Dagster → pipeline orchestration
- PyPDF / pdfminer → PDF text extraction
- LangChain / custom chunker → document chunking
- SentenceTransformers / HuggingFace → embeddings
- SQLite → local document store
- ChromaDB → vector store

## Setup and Run
Create `.env` file with following values:
```
DAGSTER_HOME
DOCUMENT_STORE_PATH
VECTOR_STORE_COLLECTION
VECTOR_STORE_PATH
GOOGLE_API_KEY
```

```bash
uv venv
source .venv/bin/activate
uv sync

python3 src/rag/download_models/download_emb_model.py

dagster dev
```
Go to [Dagster UI](http://127.0.0.1:3000/)


## Retriever

To run queries, run:

```bash
python3 src/rag/retrieval/retriever.py 
```
