#!/usr/bin/env python3
"""
ArXiv Paper Fetcher with Layout-Aware Chunking and Qdrant Storage

Workflow:
1. Extract keywords from query using glm-5.2:cloud via Ollama
2. Search ArXiv API with keywords (relevance + recent)
3. Download PDFs from ArXiv
4. Chunk PDFs using layout-aware chunking (headings-based)
   Fallback: recursive character chunking
5. Store chunks in Qdrant vector DB with metadata
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from collections import Counter
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

import fitz  # pymupdf
import fitz.utils

fitz.TOOLS.mupdf_warnings(False)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
KEYWORD_MODEL = "glm-5.2:cloud"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PDF_DIR = Path(__file__).parent / "downloaded_papers"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_PATH = Path(__file__).parent / "qdrant_storage"
COLLECTION_PREFIX = "papers"
TOTAL_PAPERS = 25
TOP_RELEVANT_RATIO = 0.8
RECENT_RATIO = 0.2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADING_FONT_THRESHOLD = 1.15
MIN_SECTIONS_FOR_LAYOUT = 3
MIN_CHARS_FOR_LAYOUT_CHECK = 5000
ARXIV_RATE_LIMIT_INTERVAL = 3
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
        "Extract the 5-10 most important technical search terms from the following research query. "
        "These terms will be used to search for academic papers on ArXiv.\n"
        "Rules:\n"
        "- Return ONLY a space-separated list of keywords\n"
        "- No explanation, no punctuation, no bullet points\n"
        "- Prefer single words or short phrases (1-2 words each)\n"
        "- Focus on the core topic, not peripheral details\n"
        "- Do not repeat keywords\n\n"
        f"Query: {query}\n\nKeywords:"
    )
    last_error = None
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": KEYWORD_MODEL, "prompt": prompt, "stream": False},
                timeout=120,
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
# Step 2: ArXiv API
# =============================================================================

ARXIV_BASE = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _arxiv_search(keywords: str, sort_by: str, sort_order: str, limit: int, progress_callback=None) -> list[dict]:
    papers = []
    start = 0
    batch_size = min(limit, 100)

    while len(papers) < limit:
        if progress_callback:
            progress_callback("fetching_papers", f"Searching ArXiv (fetched {len(papers)}/{limit} papers)...")
        params = {
            "search_query": f"all:{keywords}",
            "start": start,
            "max_results": batch_size,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        resp = requests.get(ARXIV_BASE, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        for entry in entries:
            arxiv_url = entry.find("atom:id", ATOM_NS).text.strip()
            arxiv_id = arxiv_url.split("/abs/")[-1]

            title_el = entry.find("atom:title", ATOM_NS)
            title = " ".join(title_el.text.split()) if title_el is not None else "Untitled"

            summary_el = entry.find("atom:summary", ATOM_NS)
            abstract = " ".join(summary_el.text.split()) if summary_el is not None else ""

            published_el = entry.find("atom:published", ATOM_NS)
            year = ""
            if published_el is not None:
                year = published_el.text[:4]

            authors = []
            for author in entry.findall("atom:author", ATOM_NS):
                name_el = author.find("atom:name", ATOM_NS)
                if name_el is not None:
                    authors.append({"name": name_el.text.strip()})

            pdf_link = ""
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href")
                    break

            papers.append({
                "paperId": arxiv_id,
                "title": title,
                "abstract": abstract,
                "year": year,
                "externalIds": {"ArXiv": arxiv_id},
                "openAccessPdf": {"url": pdf_link} if pdf_link else None,
                "publicationDate": published_el.text if published_el is not None else "",
                "authors": authors,
                "citationCount": 0,
            })

        start += len(entries)
        if len(entries) < batch_size:
            break
        time.sleep(ARXIV_RATE_LIMIT_INTERVAL)

    return papers[:limit]


def fetch_papers(keywords: str, progress_callback=None) -> list[dict]:
    top_n = math.ceil(TOTAL_PAPERS * TOP_RELEVANT_RATIO)
    recent_n = math.ceil(TOTAL_PAPERS * RECENT_RATIO)

    if progress_callback:
        progress_callback("fetching_papers", "Searching for most relevant papers...")
    relevant = _arxiv_search(keywords, sort_by="relevance", sort_order="descending", limit=top_n * 3, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("fetching_papers", "Searching for recent papers...")
    time.sleep(ARXIV_RATE_LIMIT_INTERVAL)
    recent = _arxiv_search(keywords, sort_by="submittedDate", sort_order="descending", limit=recent_n * 3, progress_callback=progress_callback)

    seen = set()
    merged = []
    for p in relevant + recent:
        pid = p.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            merged.append(p)

    relevant_arxiv = [p for p in merged if p in relevant][:top_n]
    recent_arxiv = [p for p in merged if p in recent][:recent_n]
    final = relevant_arxiv + recent_arxiv

    log.info(f"ArXiv papers: {len(final)} (relevant: {len(relevant_arxiv)}, recent: {len(recent_arxiv)})")
    return final


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

    ext_ids = paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    if not arxiv_id:
        log.warning(f"  No ArXiv ID for: {title[:80]}")
        return None

    pdf_url = paper.get("openAccessPdf", {})
    if pdf_url and isinstance(pdf_url, dict):
        url = pdf_url.get("url", "")
    else:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if not url:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    log.info(f"  Downloading ArXiv PDF: {url}")
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) < 1000:
            log.warning(f"  ArXiv PDF too small ({len(content)} bytes), skipping")
            return None
        filepath.write_bytes(content)
        log.info(f"  Saved: {filepath.name}")
        return filepath
    except Exception as e:
        log.warning(f"  ArXiv PDF failed: {e}")
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
# Step 5: Embeddings via Hugging Face Inference API
# =============================================================================

_hf_client = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found in .env file")
        _hf_client = InferenceClient(token=token)
    return _hf_client


def get_embedding(text: str, max_retries: int = 5) -> list[float]:
    text = text[:8000]
    last_error = None
    for attempt in range(max_retries):
        try:
            client = _get_hf_client()
            result = client.feature_extraction(text, model=EMBED_MODEL)
            if hasattr(result, "tolist"):
                result = result.tolist()
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                return result[0]
            if isinstance(result, list) and isinstance(result[0], (int, float)):
                return result
            raise ValueError(f"Unexpected embedding format: {type(result)}")
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"  Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def get_embeddings_batch(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    texts = [t[:8000] for t in texts]
    last_error = None
    for attempt in range(max_retries):
        try:
            client = _get_hf_client()
            result = client.feature_extraction(texts, model=EMBED_MODEL)
            if hasattr(result, "tolist"):
                result = result.tolist()
            if isinstance(result, list) and len(result) == len(texts) and isinstance(result[0], list):
                return result
            raise ValueError(f"Unexpected batch embedding format: {type(result)}")
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"  Batch embedding attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


# =============================================================================
# Step 6: Qdrant Storage
# =============================================================================

def _get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))


def init_qdrant(session_id: str) -> tuple[QdrantClient, str]:
    client = _get_qdrant_client()
    collection_name = f"{COLLECTION_PREFIX}_{session_id}"

    test_emb = get_embedding("dimension test")
    dim = len(test_emb)
    log.info(f"Embedding dimension: {dim}")

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info(f"Created collection: {collection_name}")

    return client, collection_name


def clear_session(session_id: str):
    """Delete the Qdrant collection for a session."""
    client = _get_qdrant_client()
    collection_name = f"{COLLECTION_PREFIX}_{session_id}"
    try:
        client.delete_collection(collection_name)
        log.info(f"Cleared collection: {collection_name}")
    except Exception:
        pass


def store_chunks(
    client: QdrantClient, collection_name: str, paper: dict, pdf_path: Path, chunks: list[dict]
):
    paper_id = paper.get("paperId") or hashlib.md5(pdf_path.name.encode()).hexdigest()
    title = paper.get("title", pdf_path.stem)
    year = paper.get("year") or ""
    authors = ", ".join(
        [a.get("name", "") for a in (paper.get("authors") or [])[:3]]
    )

    texts = [c["text"] for c in chunks]
    try:
        embeddings = get_embeddings_batch(texts)
    except Exception as e:
        log.warning(f"  Batch embedding failed ({e}), falling back to per-chunk...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embeddings.append(get_embedding(chunk["text"]))
            except Exception as e2:
                log.warning(f"  Skipping chunk {i}: embedding failed: {e2}")
                embeddings.append(None)

    points = []
    skipped = 0
    for i, chunk in enumerate(chunks):
        if i >= len(embeddings) or embeddings[i] is None:
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

        points.append(PointStruct(id=point_id, vector=embeddings[i], payload=payload))

    if points:
        client.upsert(collection_name=collection_name, points=points)
    log.info(f"  Stored {len(points)} chunks" + (f" ({skipped} skipped)" if skipped else ""))


# =============================================================================
# Main
# =============================================================================

def run_pipeline(query: str, session_id: str = "default", progress_callback=None):
    """Run the full fetch-and-index pipeline for a given query and session."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    clear_session(session_id)

    def _progress(step, detail=""):
        log.info(f"Progress: {step} - {detail}")
        if progress_callback:
            progress_callback(step, detail)

    _progress("extracting_keywords", "Extracting keywords from your query...")
    keywords = extract_keywords(query)
    _progress("extracting_keywords", f"Keywords: {keywords}")

    _progress("fetching_papers", "Searching ArXiv for relevant papers...")
    papers = fetch_papers(keywords, progress_callback=progress_callback)
    if not papers:
        _progress("error", "No papers found!")
        return
    _progress("fetching_papers", f"Found {len(papers)} papers on ArXiv")

    _progress("downloading_pdfs", f"Downloading PDFs (0/{len(papers)})...")
    downloaded = []
    for i, paper in enumerate(papers):
        title = paper.get("title", "Unknown")[:80]
        citations = paper.get("citationCount", 0)
        year = paper.get("year", "?")
        log.info(f"[{i+1:2d}/{len(papers)}] ({year}, {citations} cites) {title}")
        path = download_pdf(paper)
        if path:
            downloaded.append((paper, path))
        _progress("downloading_pdfs", f"Downloading PDFs ({len(downloaded)}/{len(papers)})...")
        time.sleep(0.5)

    if not downloaded:
        _progress("error", "No PDFs downloaded!")
        return
    _progress("downloading_pdfs", f"Downloaded {len(downloaded)} PDFs")

    _progress("initializing_qdrant", "Initializing vector database...")
    client, collection_name = init_qdrant(session_id)

    _progress("chunking_storing", f"Chunking and embedding papers (0/{len(downloaded)})...")
    for i, (paper, pdf_path) in enumerate(downloaded):
        log.info(f"[{i+1:2d}/{len(downloaded)}] {pdf_path.name}")
        chunks = chunk_pdf(pdf_path)
        store_chunks(client, collection_name, paper, pdf_path, chunks)
        _progress("chunking_storing", f"Chunking and embedding papers ({i+1}/{len(downloaded)})...")
        time.sleep(0.3)

    _progress("done", f"All done! {len(downloaded)} papers indexed.")
    log.info("=" * 60)
    log.info("Done! All papers processed and stored in Qdrant.")
    log.info(f"PDFs saved to: {PDF_DIR}")
    log.info(f"Qdrant storage: {QDRANT_PATH}")


def add_papers_to_collection(query: str, session_id: str, count: int = 3, progress_callback=None):
    """Fetch N new papers and add them to an existing Qdrant collection without clearing it."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    def _progress(step, detail=""):
        log.info(f"Background refresh: {step} - {detail}")
        if progress_callback:
            progress_callback(step, detail)

    _progress("extracting_keywords", "Extracting keywords for background refresh...")
    keywords = extract_keywords(query)
    _progress("extracting_keywords", f"Keywords: {keywords}")

    _progress("fetching_papers", f"Searching ArXiv for {count} new papers...")
    papers = fetch_papers(keywords, progress_callback=progress_callback)
    if not papers:
        _progress("error", "No papers found for background refresh!")
        return

    papers = papers[:count]

    _progress("downloading_pdfs", f"Downloading PDFs (0/{len(papers)})...")
    downloaded = []
    for i, paper in enumerate(papers):
        path = download_pdf(paper)
        if path:
            downloaded.append((paper, path))
        _progress("downloading_pdfs", f"Downloading PDFs ({len(downloaded)}/{len(papers)})...")
        time.sleep(0.5)

    if not downloaded:
        _progress("error", "No PDFs downloaded for background refresh!")
        return

    client, collection_name = init_qdrant(session_id)

    _progress("chunking_storing", f"Chunking and embedding new papers (0/{len(downloaded)})...")
    for i, (paper, pdf_path) in enumerate(downloaded):
        log.info(f"  Background: {pdf_path.name}")
        chunks = chunk_pdf(pdf_path)
        store_chunks(client, collection_name, paper, pdf_path, chunks)
        _progress("chunking_storing", f"Chunking and embedding new papers ({i+1}/{len(downloaded)})...")
        time.sleep(0.3)

    _progress("done", f"Background refresh complete! Added {len(downloaded)} new papers.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)
    run_pipeline(query)
