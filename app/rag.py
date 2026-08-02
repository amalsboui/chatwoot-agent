"""RAG layer: chunking, embedding, vector storage and retrieval.
Uses a local sentence-transformers model for embeddings and Chroma as the vector store, persisted to disk so it survives
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


def ingest_directory(directory: str) -> int:
    #Ingest the md/txt policy files in directory into the vector store
    files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True) + glob.glob(
        os.path.join(directory, "**/*.txt"), recursive=True
    )
    collection = _get_collection()
    total_chunks = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_text(text)
        if not chunks:
            continue
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": os.path.basename(path)} for _ in chunks]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
    return total_chunks


def retrieve(query: str, k: int | None = None) -> List[dict]:
    # Return top-k relevant chunks for a query
    k = k or settings.top_k
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source", "unknown"), "score": 1 - dist})
    return hits