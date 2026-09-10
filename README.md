# Lemma

Lemma is a lightweight Python project for experimenting with retrieval-augmented generation workflows over research documents and arXiv-style data.

## Overview

This repository contains a minimal starter structure for:

- downloading and preparing document data
- indexing research text into chunks
- retrieving relevant passages with embeddings
- grounding a language model answer in retrieved context

## Project structure

- `main.py` - general project entry point
- `src/basic_rag.py` - initial RAG implementation using chunking, embedding similarity, and grounded generation
- `data/` - datasets and corpus assets
- `scripts/` - helper scripts for data collection
- `pyproject.toml` - Python project configuration and dependencies

## Getting started

1. Install dependencies:

```bash
uv sync
```

2. Download the required arXiv source documents:

```bash
python scripts/get_arxiv_pdfs.py
```

3. Start a local Ollama server with the required models available (the starter script uses `nomic-embed-text` and `llama3.1`):

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

4. Run the initial RAG prototype:

```bash
python src/basic_rag.py
```

## Notes

This is a starter repository intended to be expanded into a more complete retrieval and generation pipeline.
