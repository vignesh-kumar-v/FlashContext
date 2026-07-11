from celery import Celery
from backend.config import REDIS_BROKER_URL, REDIS_BACKEND_URL

celery_app = Celery(
    "flashcontext",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BACKEND_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["backend"])
