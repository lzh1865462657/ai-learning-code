from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
#私有知识库文档,后续可替代成业务文档
knowledge_text = """
1. FastAPI是Python高性能AI后端框架，自带/docs可视化接口调试页面，适合封装大模型问答服务。
2. Ollama可本地离线运行开源大模型，默认占用11434端口，无需联网、无调用计费。
3. Prompt工程三大核心：定义模型角色、约束输出格式、限制信息来源，用于消除大模型幻觉。
4. RAG检索增强生成流程：文档分割→文本向量化入库→语义检索匹配资料给LLM生成答案。
5. Chroma是轻量本地向量数据库，无需额外部署服务，向量持久化保存在本地文件夹。
6. LangChain封装RAG全套工具，简化文档加载、分割、向量存储、大模型调用重复代码。
7. Linux常用部署命令：nohup后台运行程序、tail查看日志、虚拟环境隔离项目依赖。
"""

#文本分割
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
text_chunks=text_splitter.split_text(knowledge_text)

embedding=OllamaEmbeddings(model="nomic-embed-text")
#初始化向量库，持久化存储到本地chroma_db文件夹
vector_store=Chroma.from_texts(
    texts=text_chunks,
    embedding=embedding,
    persist_directory=str(BASE_DIR / "chroma_db")
)
print(f"文档入库完成，向量数据保存在 {BASE_DIR / 'chroma_db'}")
