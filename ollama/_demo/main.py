from fastapi  import FastAPI
import requests
app=FastAPI(title="本地OLLAMA问答服务")
OLLAMA_URL="http://localhost:11434/api/genrate"
MODEL="qwen:7b"
def llm_call(question:str):
    prompt=f"""
    "你是AI后端开发工程师，回答精简专业，无相关知识直接说明。
提问:{question}
    """
    data={"model":MODEL,"prompt":prompt,"stream":False}
    r=requests.post(OLLAMA_URL,json=data)
    return r.json()["response"]
@app.post("/chat")
def chat(quesstion:str):
    ans=llm_call(quesstion)
    return{"question":question,"answer":ans}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8001)