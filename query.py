#!/usr/bin/env python3
"""
RAG Query Script: Retrieve chunks from Qdrant and generate answers via glm-5.2:cloud.

Workflow:
1. Take user query
2. Check if Qdrant has relevant chunks; if not, run fetch_and_index pipeline
3. Embed query via Hugging Face Inference API
4. Search Qdrant for top-k relevant chunks
5. Build prompt with retrieved context
6. Generate response via glm-5.2:cloud (Ollama)
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
import requests

load_dotenv()

# --- Configuration ---
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
LLM_MODEL = "glm-5.2:cloud"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_PATH = Path(__file__).parent / "qdrant_storage"
COLLECTION_PREFIX = "papers"
TOP_K = 5
# --- End Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_hf_client = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found in .env file")
        _hf_client = InferenceClient(token=token)
    return _hf_client


def embed_query(text: str) -> list[float]:
    client = _get_hf_client()
    result = client.feature_extraction(text, model=EMBED_MODEL)
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    if isinstance(result, list) and isinstance(result[0], (int, float)):
        return result
    raise ValueError(f"Unexpected embedding format: {type(result)}")


def _get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))


def search_qdrant(query: str, session_id: str = "default", top_k: int = TOP_K) -> list[dict]:
    client = _get_qdrant_client()
    collection_name = f"{COLLECTION_PREFIX}_{session_id}"
    emb = embed_query(query)
    results = client.query_points(
        collection_name, query=emb, limit=top_k, with_payload=True
    )
    chunks = []
    for r in results.points:
        chunks.append({
            "score": r.score,
            "title": r.payload.get("title", "Unknown"),
            "section": r.payload.get("source_section", "Unknown"),
            "text": r.payload.get("text", ""),
        })
    return chunks


def detect_topic_shift(chunks: list[dict]) -> bool:
    """Return True if the top chunk scores are low, indicating a topic shift."""
    if not chunks:
        return True
    top_scores = [c["score"] for c in chunks[:3]]
    avg = sum(top_scores) / len(top_scores)
    return avg < 0.25


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}] Title: {c['title']}\n"
            f"Section: {c['section']}\n"
            f"Content: {c['text']}"
        )
    context = "\n\n".join(context_parts)

    return (
        "You are a helpful research assistant. Answer the user's question using your "
        "own knowledge and expertise. The provided context below contains relevant "
        "research papers — use it to enrich your answer when it helps, but do not "
        "limit yourself to it. If the user asks for code, write it. If they ask for "
        "explanations, provide them. Be thorough and helpful.\n\n"
        "IMPORTANT: Never start your answer with phrases like 'Based on the provided context' "
        "or 'According to the context'. Just answer directly and naturally.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_response(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate_response_stream(prompt: str):
    """Stream tokens from Ollama as they are generated."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue


def _qdrant_has_data(session_id: str = "default") -> bool:
    try:
        client = _get_qdrant_client()
        collection_name = f"{COLLECTION_PREFIX}_{session_id}"
        info = client.get_collection(collection_name)
        return info.points_count > 0
    except Exception:
        return False


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ").strip()

    if not query:
        print("No query provided.")
        sys.exit(1)

    log.info(f"Query: {query}")

    if not _qdrant_has_data():
        log.info("First query — populating DB with relevant papers (this will take a moment)...")
        from fetch_and_index import run_pipeline
        run_pipeline(query)
    else:
        log.info("DB already populated — searching directly...")

    log.info("Searching Qdrant for relevant chunks...")
    chunks = search_qdrant(query)
    if not chunks:
        print("No relevant chunks found in the database.")
        sys.exit(1)

    log.info(f"Found {len(chunks)} relevant chunks:")
    for i, c in enumerate(chunks):
        log.info(f"  [{i+1}] {c['title'][:60]} (score: {c['score']:.4f})")

    log.info("Building prompt and generating response...")
    prompt = build_prompt(query, chunks)
    response = generate_response(prompt)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response)
    print("=" * 60)

    print("\nSources:")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c['title']} (score: {c['score']:.4f})")


if __name__ == "__main__":
    main()
