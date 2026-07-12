import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from backend.celery_app import celery_app
from backend.config import REDIS_BROKER_URL
from fetch_and_index import run_pipeline
from query import search_qdrant, build_prompt, generate_response_stream


def _get_redis():
    return redis.from_url(REDIS_BROKER_URL)


@celery_app.task(bind=True, name="backend.tasks.run_scraping_job")
def run_scraping_job(self, query: str, session_id: str = "default"):
    def on_progress(step, detail):
        self.update_state(state="PROGRESS", meta={"step": step, "detail": detail})

    run_pipeline(query, session_id=session_id, progress_callback=on_progress)
    return {"status": "completed", "query": query, "session_id": session_id}


@celery_app.task(bind=True, name="backend.tasks.run_rag_query")
def run_rag_query(self, query: str, session_id: str = "default"):
    self.update_state(state="PROGRESS", meta={"step": "searching_qdrant", "detail": "Searching knowledge base..."})
    chunks = search_qdrant(query, session_id=session_id)
    if not chunks:
        return {"status": "no_results", "query": query, "answer": None, "sources": []}

    sources = [{"title": c["title"], "score": c["score"]} for c in chunks]

    self.update_state(state="PROGRESS", meta={"step": "generating_response", "detail": "Generating answer..."})
    prompt = build_prompt(query, chunks)

    r = _get_redis()
    channel = f"stream:{session_id}"
    full_response = ""

    for token in generate_response_stream(prompt):
        full_response += token
        r.publish(channel, json.dumps({"token": token}))

    r.publish(channel, json.dumps({"done": True, "answer": full_response, "sources": sources}))

    return {"status": "completed", "query": query, "answer": full_response, "sources": sources}
