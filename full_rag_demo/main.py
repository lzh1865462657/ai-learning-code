from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import configure_logging
from document_loader import DocumentProcessingError, index_document
from llm_service import LLMServiceError, answer_question, stream_answer
from vector_store import VectorStoreError


app = FastAPI(title="离线私有知识库 RAG 问答系统", version="1.2.0")
configure_logging()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, description="要向知识库提出的问题")


class DocumentRequest(BaseModel):
    file_path: str = Field(description="服务器本地的 TXT 或 PDF 文档路径")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/index")
def index_local_document(request: DocumentRequest) -> dict[str, str | int]:
    try:
        chunk_count = index_document(request.file_path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DocumentProcessingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except VectorStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "message": "文档入库完成",
        "file_path": request.file_path,
        "chunk_count": chunk_count,
    }


@app.post("/knowledge_chat")
def knowledge_chat(request: QuestionRequest) -> dict[str, str]:
    try:
        answer = answer_question(request.question)
    except LLMServiceError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"question": request.question, "answer": answer}


@app.post("/stream_chat")
def stream_chat(request: QuestionRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(request.question),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
