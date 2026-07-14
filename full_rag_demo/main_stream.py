from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from stream_rag_core import stream_rag_answer
app=FastAPI(title="流式RAG私有知识库系统")
@app.post("/stream_chat")
def stream_chat(question:str):
    generator=stream_rag_answer(question)
    return StreamingResponse(generator,media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
