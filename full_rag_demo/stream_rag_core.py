import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:7b"
EMBEDDING_MODEL = "nomic-embed-text"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"


def load_chroma_db():
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embedding,
    )


def get_ref_context(user_question:str,top_k=2):
    db=load_chroma_db()
    search_result=db.similarity_search(user_question,k=top_k)
    context="\n".join([doc.page_content for doc in search_result])
    return context
#生成函数器，流式返回结果
def stream_rag_answer(user_question:str):
    ref_info=get_ref_context(user_question)
    prompt=f"""
你是私有知识库问答助手，严格遵循规则：
1. 只能使用【参考知识库】内容回答，无相关内容直接回复「暂无对应知识库资料」；
2. 禁止编造知识点，回答简洁易懂。

【参考知识库】
{ref_info}

【用户提问】
{user_question}
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }
#开启流式请求
    resp=requests.post(
        OLLAMA_API,
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line :
            data=json.loads(line)
            token=data.get("response","")
            yield token
#bendiceshi
if __name__=="__main__":
    stream=stream_rag_answer("RAG有哪些步骤")
    for word in stream:
        print(word,end="",flush=True)
