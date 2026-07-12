import asyncio
import json
import redis.asyncio as redis_lib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from celery.result import AsyncResult

from backend.tasks import run_scraping_job, run_rag_query
from backend.config import REDIS_BROKER_URL

app = FastAPI(title="FlashContext", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class TaskResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    progress: dict | None = None


# ── HTTP (kept for backward compatibility) ──────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scrape", response_model=TaskResponse)
def start_scrape(req: QueryRequest):
    task = run_scraping_job.delay(req.query, session_id=req.session_id)
    return TaskResponse(task_id=task.id, status="PENDING")


@app.get("/scrape/{task_id}", response_model=TaskStatusResponse)
def get_scrape_status(task_id: str):
    result = AsyncResult(task_id, app=run_scraping_job)
    response = TaskStatusResponse(task_id=task_id, status=result.state)
    if result.state == "PROGRESS":
        response.progress = result.info
    elif result.state == "SUCCESS":
        response.result = result.result
    elif result.state == "FAILURE":
        response.result = {"error": str(result.info)}
    return response


@app.post("/query", response_model=TaskResponse)
def start_query(req: QueryRequest):
    task = run_rag_query.delay(req.query, session_id=req.session_id)
    return TaskResponse(task_id=task.id, status="PENDING")


@app.get("/query/{task_id}", response_model=TaskStatusResponse)
def get_query_status(task_id: str):
    result = AsyncResult(task_id, app=run_rag_query)
    response = TaskStatusResponse(task_id=task_id, status=result.state)
    if result.state == "PROGRESS":
        response.progress = result.info
    elif result.state == "SUCCESS":
        response.result = result.result
    elif result.state == "FAILURE":
        response.result = {"error": str(result.info)}
    return response


# ── WebSocket ────────────────────────────────────────────────────────

async def _poll_task(task_id: str, task_func, websocket: WebSocket):
    last_state = None
    while True:
        result = await asyncio.to_thread(AsyncResult, task_id, app=task_func)
        state = result.state

        if state != last_state or state == "PROGRESS":
            last_state = state
            msg = {"task_id": task_id, "status": state}
            if state == "PROGRESS" and result.info:
                msg["progress"] = result.info
            elif state == "SUCCESS":
                msg["result"] = result.result
                await websocket.send_json(msg)
                return
            elif state == "FAILURE":
                msg["result"] = {"error": str(result.info)}
                await websocket.send_json(msg)
                return
            await websocket.send_json(msg)

        await asyncio.sleep(1)


async def _stream_query(task_id: str, session_id: str, websocket: WebSocket):
    r = redis_lib.from_url(REDIS_BROKER_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"stream:{session_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data = json.loads(message["data"])

            if "token" in data:
                await websocket.send_json({"token": data["token"]})
            elif "progress" in data:
                await websocket.send_json({"status": "PROGRESS", "progress": data["progress"]})
            elif data.get("done"):
                await websocket.send_json({
                    "status": "SUCCESS",
                    "result": {
                        "status": "completed",
                        "answer": data["answer"],
                        "sources": data["sources"],
                    },
                })
                break
    finally:
        await pubsub.unsubscribe()
        await pubsub.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    msg_type = data.get("type")
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")

    if not query:
        await websocket.send_json({"status": "ERROR", "result": {"error": "No query provided"}})
        await websocket.close()
        return

    if msg_type == "scrape":
        task = run_scraping_job.delay(query, session_id=session_id)
        await websocket.send_json({"task_id": task.id, "status": "PENDING"})
        await _poll_task(task.id, run_scraping_job, websocket)

    elif msg_type == "query":
        task = run_rag_query.delay(query, session_id=session_id)
        await websocket.send_json({"task_id": task.id, "status": "PENDING"})
        await _stream_query(task.id, session_id, websocket)

    else:
        await websocket.send_json({"status": "ERROR", "result": {"error": f"Unknown type: {msg_type}"}})

    await websocket.close()
