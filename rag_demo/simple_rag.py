from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import requests
#"文档"
doc = """Prompt工程用于规范大模型输出，减少幻觉。
写Prompt要先定义角色、约束输出格式、限定信息来源。
RAG系统会把检索到的文档片段拼接进Prompt，让模型基于资料回答。
Ollama支持本地离线部署通义千问、Llama等开源大模型。
LangChain封装了文档加载、分割、向量库、LLM调用全套工具。"""
splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
chunks=splitter.split_text(doc)

embed=OllamaEmbeddings(model="nomic-embed-text")
chunk_vectors=[embed.embed_query(c) for c in chunks]

OLLAMA_URL="http://localhost:11434/api/generate"
def llm_answer(question,context):
    prompt=f"""
    "仅根据下面参考资料回答问题，不知道就直接说无相关信息，禁止编造。
参考资料：{context}
用户提问：{question}
    """
    payload={"model":"qwen:7b","prompt":prompt,"stream":False}
    res=requests.post(OLLAMA_URL,json=payload,timeout=120)
    res.raise_for_status()
    return res.json()["response"]

#模拟提问
user_q ="Prompt在RAG里有什么作用？"
ans=llm_answer(user_q,"\n".join(chunks))
print("RAG回答:\n",ans)
