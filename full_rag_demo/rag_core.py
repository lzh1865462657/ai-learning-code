from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:7b"
EMBEDDING_MODEL = "nomic-embed-text"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"

#读取刚才存好的向量库
def load_chroma_db():
    emb_model=OllamaEmbeddings(model=EMBEDDING_MODEL)
    db=Chroma(persist_directory=str(DB_PATH),embedding_function=emb_model)
    return db
def get_ref_context(user_question:str,top_k=2):
    db=load_chroma_db()
    search_result=db.similarity_search(user_question,k=top_k)
    context="\n".join([doc.page_content for doc in search_result])
    return context
#完整问答流程：
def rag_answer(user_question:str):
    ref_info=get_ref_context(user_question)
    prompt_template=f"""你是企业私有知识库专属问答助手，严格遵守规则：
1. 只能使用下方【参考知识库】内容回答，无相关内容直接回复「暂无对应知识库资料」；
2. 禁止编造不存在的技术知识点，不拓展无关内容；
3. 回答简洁分点，语言专业清晰。
(参考数据库)
{ref_info}
（用户提问）
{user_question}
"""
    payload={
        "model":MODEL_NAME,
        "prompt":prompt_template,
        "stream":False
    }
    response=requests.post(OLLAMA_API,json=payload,timeout=120)
    response.raise_for_status()
    return response.json()["response"]

#本地测试入口
if __name__=="__main__":
    result=rag_answer("RAG完整执行步骤是什么")
    print("__RAG问答结果__")
    print(result)
