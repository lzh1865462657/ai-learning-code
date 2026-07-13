from fastapi import FastAPI
from rag_core import rag_answer
#创建王爷服务
app=FastAPI(title="离线私有知识库RAG问答系统", version="1.0")
#网页提问接口
@app.post("/knowledge_chat")
def knowledge_qa(question:str):
    answer=rag_answer(question)
    return{
        "用户提问":question,
        "知识库问答":answer
    }
#启动服务
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)