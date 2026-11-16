"""FastAPI entry point; start with: python -m uvicorn api:app --port 8000."""
import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel, Field

from embeddings.config import EMBEDDING_CONFIG
from fast_rag import FastRag, Gemini

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")


def open_collection(cfg):
    import chromadb
    manifest = json.loads(Path(cfg.persist_dir, "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("embedding_model") != cfg.model or
            manifest.get("dimensions") != cfg.output_dimensionality or
            manifest.get("fake_embeddings")):
        raise RuntimeError("Index embedding model/dimensions do not match. Rebuild the index before starting.")
    return chromadb.PersistentClient(path=cfg.persist_dir).get_collection(cfg.collection_name)


@asynccontextmanager
async def lifespan(app):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in hcl-tech-retrival/.env.")
    model = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite").strip()
    cfg = replace(EMBEDDING_CONFIG, persist_dir=str(ROOT / EMBEDDING_CONFIG.persist_dir))
    collection = await asyncio.to_thread(open_collection, cfg)
    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        headers={"x-goog-api-key": key},
        timeout=httpx.Timeout(10, connect=3, pool=2),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    ) as client:
        app.state.rag = FastRag(Gemini(client, model, cfg), collection)
        logger.info("Ready: model=%s, %d chunks, total request deadline=20s, automatic retries=0",
                    app.state.rag.gemini.model, app.state.rag.count)
        yield


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=100)


def arguments(request):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    return question, [m.model_dump() for m in request.history[-4:]]


@app.get("/health")
def health():
    # Local readiness only; this does not call Gemini or validate API-key access.
    return {"status": "running", "build": "fast-rag-stream-v2",
            "model": app.state.rag.gemini.model, "chunks": app.state.rag.count,
            "request_deadline_seconds": 20, "streaming": True}


@app.post("/answer/stream")
async def answer_stream(request: ChatRequest):
    question, history = arguments(request)

    async def body():
        async with aclosing(app.state.rag.events(question, history)) as events:
            async for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(body(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.post("/answer")
async def answer(request: ChatRequest):
    """Compatibility endpoint for clients expecting a single JSON response."""
    question, history = arguments(request)
    async with aclosing(app.state.rag.events(question, history)) as events:
        async for event in events:
            if event["type"] == "error":
                raise HTTPException(event["status"], event["error"])
            if event["type"] == "done":
                return event
    raise HTTPException(502, "Answer stream ended unexpectedly.")
