# Lemma

Lemma is a lightweight Python project for experimenting with retrieval-augmented generation workflows over research documents and arXiv-style data.

## Overview

This repository contains a minimal starter structure for:

- downloading and preparing document data
- building a simple RAG prototype over research papers

## Project structure

- `main.py` - entry point for the project
- `data/` - datasets and corpus assets
- `scripts/` - helper scripts for data collection
- `pyproject.toml` - Python project configuration and dependencies

## Getting started

Before running the project, fetch the required arXiv documents:

```bash
python scripts/get_arxiv_pdfs.py
```

This script downloads the documents needed for the local dataset used by the project.

## Notes

This is a starter repository and is intended to be expanded as the project grows.
