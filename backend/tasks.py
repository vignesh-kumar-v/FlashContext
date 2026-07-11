import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.celery_app import celery_app
from fetch_and_index import run_pipeline
from query import search_qdrant, build_prompt, generate_response


@celery_app.task(bind=True, name="backend.tasks.run_scraping_job")
def run_scraping_job(self, query: str, session_id: str = "default"):
    self.update_state(state="PROGRESS", meta={"step": "extracting_keywords"})
    run_pipeline(query, session_id=session_id)
    return {"status": "completed", "query": query, "session_id": session_id}


@celery_app.task(bind=True, name="backend.tasks.run_rag_query")
def run_rag_query(self, query: str, session_id: str = "default"):
    self.update_state(state="PROGRESS", meta={"step": "searching_qdrant"})
    chunks = search_qdrant(query, session_id=session_id)
    if not chunks:
        return {"status": "no_results", "query": query, "answer": None, "sources": []}

    self.update_state(state="PROGRESS", meta={"step": "generating_response"})
    prompt = build_prompt(query, chunks)
    answer = generate_response(prompt)

    sources = [
        {"title": c["title"], "score": c["score"]} for c in chunks
    ]

    return {"status": "completed", "query": query, "answer": answer, "sources": sources}
