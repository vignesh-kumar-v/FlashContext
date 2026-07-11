#!/usr/bin/env python3
"""
Semantic Scholar Paper Fetcher with Layout-Aware Chunking and Qdrant Storage

Workflow:
1. Extract keywords from query using qwen3.5:9b-mlx via Ollama
2. Search Semantic Scholar API with keywords
3. Fetch papers: 80% top-cited, 20% most recent
4. Download PDFs to local folder
5. Chunk PDFs using layout-aware chunking (headings-based)
   Fallback: recursive character chunking
6. Store chunks in Qdrant vector DB with metadata
"""

import os
import sys
import json
import math
import time
import uuid
import hashlib
import logging
import requests
from pathlib import Path
from typing import Optional
from collections import Counter

import fitz  # pymupdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
QUERY = "I want to build a GPT style transformer using Muon Optimizer"
OLLAMA_BASE = "http://localhost:11434"
KEYWORD_MODEL = "qwen3.5:9b-mlx"
EMBED_MODEL = "qwen3-embedding:0.6b"
PDF_DIR = Path(__file__).parent / "downloaded_papers"
QDRANT_PATH = Path(__file__).parent / "qdrant_storage"
COLLECTION_NAME = "papers"
TOTAL_PAPERS = 25
TOP_CITED_RATIO = 0.8
RECENT_RATIO = 0.2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADING_FONT_THRESHOLD = 1.15
MIN_SECTIONS_FOR_LAYOUT = 3
MIN_CHARS_FOR_LAYOUT_CHECK = 5000
# --- End Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Step 1: Keyword Extraction via Ollama
# =============================================================================

def extract_keywords(query: str) -> str:
    prompt = (
        "Extract the key technical terms and concepts from the following research query. "
        "Return ONLY a space-separated list of keywords, no explanation, no punctuation.\n\n"
        f"Query: {query}\n\nKeywords:"
    )
    last_error = None
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": KEYWORD_MODEL, "prompt": prompt, "stream": False},
                timeout=300,
            )
            resp.raise_for_status()
            keywords = resp.json()["response"].strip()
            log.info(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            last_error = e
            if attempt < 4:
                wait = 2 ** attempt
                log.warning(f"Keyword extraction attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


# =============================================================================
# Step 2: Semantic Scholar API
# =============================================================================

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,openAccessPdf,publicationDate,authors"


def _s2_search(keywords: str, sort: str, limit: int) -> list[dict]:
    papers = []
    offset = 0
    batch_size = min(limit, 100)

    while len(papers) < limit:
        params = {
            "query": keywords,
            "limit": batch_size,
            "offset": offset,
            "sort": sort,
            "fields": S2_FIELDS,
        }
        resp = requests.get(f"{S2_BASE}/paper/search", params=params, timeout=30)
        if resp.status_code == 429:
            log.warning("Rate limited (429), waiting 5s...")
            time.sleep(5)
            continue
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("data", [])
        if not batch:
            break
        papers.extend(batch)
        offset += len(batch)
        if len(batch) < batch_size:
            break
        time.sleep(1.1)

    return papers[:limit]


def fetch_papers(keywords: str) -> list[dict]:
    top_n = math.ceil(TOTAL_PAPERS * TOP_CITED_RATIO)
    recent_n = math.ceil(TOTAL_PAPERS * RECENT_RATIO)

    log.info(f"Fetching {top_n} top-cited papers...")
    cited = _s2_search(keywords, sort="citationCount:desc", limit=top_n)

    log.info(f"Fetching {recent_n} recent papers...")
    recent = _s2_search(keywords, sort="publicationDate:desc", limit=recent_n)

    seen = set()
    merged = []
    for p in cited + recent:
        pid = p.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            merged.append(p)

    log.info(f"Total unique papers: {len(merged)}")
    return merged


# =============================================================================
# Step 3: PDF Download
# =============================================================================

def download_pdf(paper: dict) -> Optional[Path]:
    paper_id = paper.get("paperId", "unknown")
    title = paper.get("title", "untitled")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    filename = f"{paper_id}_{safe_title}.pdf"
    filepath = PDF_DIR / filename

    if filepath.exists():
        log.info(f"  Already downloaded: {filepath.name}")
        return filepath

    oa = paper.get("openAccessPdf")
    if oa and oa.get("url"):
        url = oa["url"]
        log.info(f"  Downloading OA PDF: {url}")
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            filepath.write_bytes(r.content)
            log.info(f"  Saved: {filepath.name}")
            return filepath
        except Exception as e:
            log.warning(f"  OA PDF failed: {e}")

    ext_ids = paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    if arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        log.info(f"  Downloading ArXiv PDF: {url}")
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            filepath.write_bytes(r.content)
            log.info(f"  Saved: {filepath.name}")
            return filepath
        except Exception as e:
            log.warning(f"  ArXiv PDF failed: {e}")

    log.warning(f"  No PDF available for: {title[:80]}")
    return None


# =============================================================================
# Step 4: Layout-Aware Chunking
# =============================================================================

def _detect_body_font_size(page: fitz.Page) -> float:
    blocks = page.get_text("dict")["blocks"]
    sizes = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span["size"])
    if not sizes:
        return 11.0
    return Counter(sizes).most_common(1)[0][0]


def _extract_sections(doc: fitz.Document) -> list[dict]:
    sections = []
    current_heading = "Preamble"
    current_text = []

    for page_num, page in enumerate(doc):
        body_size = _detect_body_font_size(page)
        heading_threshold = body_size * HEADING_FONT_THRESHOLD

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue

            block_text = ""
            block_max_size = 0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span["text"]
                    block_max_size = max(block_max_size, span["size"])

            block_text = block_text.strip()
            if not block_text:
                continue

            is_heading = (
                block_max_size >= heading_threshold
                and len(block_text.split()) <= 15
                and len(block_text) < 150
            )

            if is_heading:
                if current_text:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_text),
                        "page": page_num + 1,
                    })
                current_heading = block_text
                current_text = []
            else:
                current_text.append(block_text)

    if current_text:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_text),
            "page": page_num + 1,
        })

    return sections


def layout_aware_chunk(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    sections = _extract_sections(doc)
    total_chars = sum(len(s["text"]) for s in sections)
    doc.close()

    if not sections:
        raise ValueError("No sections extracted from PDF")

    if (
        len(sections) < MIN_SECTIONS_FOR_LAYOUT
        and total_chars > MIN_CHARS_FOR_LAYOUT_CHECK
    ):
        raise ValueError(
            f"Only {len(sections)} section(s) for {total_chars} chars — "
            f"heading detection likely failed"
        )

    chunks = []
    for sec in sections:
        text = sec["text"]
        heading = sec["heading"]
        page = sec["page"]

        if len(text) <= CHUNK_SIZE:
            chunks.append({
                "text": text,
                "metadata": {
                    "source_section": heading,
                    "page": page,
                    "chunk_method": "layout_aware",
                },
            })
        else:
            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[i : i + CHUNK_SIZE]
                chunk_text = " ".join(chunk_words)
                if len(chunk_text) < 50:
                    continue
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_section": heading,
                        "page": page,
                        "chunk_method": "layout_aware",
                    },
                })

    return chunks


def recursive_char_chunk(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    full_text = ""
    page_offsets = []
    for page in doc:
        page_offsets.append(len(full_text))
        full_text += page.get_text() + "\n"
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for chunk_text in splitter.split_text(full_text):
        char_pos = full_text.find(chunk_text[:80]) if len(chunk_text) >= 80 else 0
        est_page = 1
        for pi, off in enumerate(page_offsets):
            if char_pos >= off:
                est_page = pi + 1

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source_section": f"Unknown / Page {est_page}",
                "page": est_page,
                "chunk_method": "recursive_char",
            },
        })

    return chunks


def chunk_pdf(pdf_path: Path) -> list[dict]:
    try:
        chunks = layout_aware_chunk(pdf_path)
        log.info(f"  Layout-aware: {len(chunks)} chunks")
        return chunks
    except Exception as e:
        log.warning(f"  Layout-aware failed: {e}")
        log.info(f"  Falling back to recursive character chunking...")
        chunks = recursive_char_chunk(pdf_path)
        log.info(f"  Recursive: {len(chunks)} chunks")
        return chunks


# =============================================================================
# Step 5: Embeddings via Ollama
# =============================================================================

def get_embedding(text: str, max_retries: int = 5) -> list[float]:
    text = text[:8000]
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"  Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


# =============================================================================
# Step 6: Qdrant Storage
# =============================================================================

def init_qdrant() -> QdrantClient:
    client = QdrantClient(path=str(QDRANT_PATH))

    test_emb = get_embedding("dimension test")
    dim = len(test_emb)
    log.info(f"Embedding dimension: {dim}")

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info(f"Created collection: {COLLECTION_NAME}")

    return client


def store_chunks(
    client: QdrantClient, paper: dict, pdf_path: Path, chunks: list[dict]
):
    paper_id = paper.get("paperId") or hashlib.md5(pdf_path.name.encode()).hexdigest()
    title = paper.get("title", pdf_path.stem)
    year = paper.get("year") or ""
    authors = ", ".join(
        [a.get("name", "") for a in (paper.get("authors") or [])[:3]]
    )

    points = []
    skipped = 0
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk["text"])
        except Exception as e:
            log.warning(f"  Skipping chunk {i}: embedding failed: {e}")
            skipped += 1
            continue
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}_{i}"))

        payload = {
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "authors": authors,
            "chunk_index": i,
            "text": chunk["text"],
            **chunk["metadata"],
        }

        points.append(PointStruct(id=point_id, vector=emb, payload=payload))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    log.info(f"  Stored {len(points)} chunks" + (f" ({skipped} skipped)" if skipped else ""))


# =============================================================================
# Main
# =============================================================================

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Step 1: Extracting keywords...")
    keywords = extract_keywords(QUERY)

    log.info("=" * 60)
    log.info("Step 2: Fetching papers from Semantic Scholar...")
    papers = fetch_papers(keywords)
    if not papers:
        log.error("No papers found!")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Step 3: Downloading PDFs...")
    downloaded = []
    for i, paper in enumerate(papers):
        title = paper.get("title", "Unknown")[:80]
        citations = paper.get("citationCount", 0)
        year = paper.get("year", "?")
        log.info(f"[{i+1:2d}/{len(papers)}] ({year}, {citations} cites) {title}")
        path = download_pdf(paper)
        if path:
            downloaded.append((paper, path))
        time.sleep(0.5)

    log.info(f"Downloaded {len(downloaded)}/{len(papers)} PDFs")
    if not downloaded:
        log.error("No PDFs downloaded!")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Step 4: Initializing Qdrant...")
    client = init_qdrant()

    log.info("=" * 60)
    log.info("Step 5: Chunking and storing...")
    for i, (paper, pdf_path) in enumerate(downloaded):
        log.info(f"[{i+1:2d}/{len(downloaded)}] {pdf_path.name}")
        chunks = chunk_pdf(pdf_path)
        store_chunks(client, paper, pdf_path, chunks)
        time.sleep(0.3)

    log.info("=" * 60)
    log.info("Done! All papers processed and stored in Qdrant.")
    log.info(f"PDFs saved to: {PDF_DIR}")
    log.info(f"Qdrant storage: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
