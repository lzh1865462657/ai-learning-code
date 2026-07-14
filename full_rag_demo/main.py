from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from document_loader import index_document
from llm_service import answer_question, stream_answer


app = FastAPI(title="离线私有知识库 RAG 问答系统", version="1.1.0")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, description="要向知识库提出的问题")


class DocumentRequest(BaseModel):
    file_path: str = Field(description="服务器本地的 TXT 或 PDF 文档路径")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/index")
def index_local_document(request: DocumentRequest) -> dict[str, str]:
    try:
        index_document(request.file_path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"message": "文档入库完成", "file_path": request.file_path}


@app.post("/knowledge_chat")
def knowledge_chat(request: QuestionRequest) -> dict[str, str]:
    return {"question": request.question, "answer": answer_question(request.question)}


@app.post("/stream_chat")
def stream_chat(request: QuestionRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(request.question),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
