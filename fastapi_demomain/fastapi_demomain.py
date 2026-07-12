from fastapi import FastAPI

# 实例化服务对象
app = FastAPI(title="大模型问答简易服务", version="1.0")

# 根路由 GET 请求
@app.get("/")
def index():
    return {"msg": "服务启动成功，访问 /docs 调试接口"}

# 问答POST接口（接收用户问题）
@app.post("/chat")
def chat(question: str):
    answer = f"收到你的提问：{question}，这是简易返回回答"
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 允许局域网/外部访问，端口8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
from pydantic import BaseModel
class ChatRep(BaseModel):
    question_list:list[str]
    user_id:int
@app.post("/batch_chat")
def batch_chat(req:ChatRep):
    res=[]
    for q in req.question_list:
        res.append({"q":q,"ans":f"回答:{q}"})
        return {"user_id":req.user_id,"result":res}
    
