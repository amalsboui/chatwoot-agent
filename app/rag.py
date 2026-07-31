"""RAG layer: chunking, embedding, vector storage and retrieval.

Uses a local sentence-transformers model for embeddings (no extra API key
needed) and Chroma as the vector store, persisted to disk so it survives
container restarts.
"""
from __future__ import annotations

import glob
import os
import uuid
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

_collection = None  # lazy-initialized so importing this module never triggers
                     # a model download / network call


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        _collection = client.get_or_create_collection(
            name="knowledge_base", embedding_function=embedder
        )
    return _collection


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return [c for c in chunks if c.strip()]

