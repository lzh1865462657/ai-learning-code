"""FastAPI application exposing the local RAG project as HTTP endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_learning.rag import TfidfRetriever, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.json"

retriever = TfidfRetriever(load_documents(KNOWLEDGE_PATH))
app = FastAPI(
    title="AI Learning RAG API",
    description="一个可离线运行的检索增强问答学习项目",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=2, ge=1, le=5)


class BatchChatRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=2, ge=1, le=5)


@app.get("/")
def index() -> dict[str, str]:
    return {"message": "服务已启动，请访问 /docs 调试接口"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "document_count": len(retriever.documents)}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    try:
        response = retriever.answer(request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"question": request.question, **response}


@app.post("/batch-chat")
def batch_chat(request: BatchChatRequest) -> dict[str, object]:
    results = [
        {"question": question, **retriever.answer(question, top_k=request.top_k)}
        for question in request.questions
    ]
    return {"count": len(results), "results": results}
