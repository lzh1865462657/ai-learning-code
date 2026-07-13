from langchain_text_splitters import RecursiveCharacterTextSplitter

long_doc="""\
"RAG检索增强生成是大模型落地最常用方案。
它解决大模型知识滞后、容易幻觉、私有数据无法读取的问题。
整体分为五大步骤：文档加载、文本分块、Embedding向量化、向量库存储、相似度检索、LLM拼接prompt回答。
Prompt工程在RAG里至关重要，必须约束模型只能使用检索到的文档片段作答，禁止编造信息。
Ollama可以本地离线运行开源大模型，搭配LangChain快速搭建私有知识库，不需要调用第三方付费API。
FastAPI可以把RAG封装成后端接口，给前端页面调用，做成完整问答系统。
"""

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunk_list=text_splitter.split_text(long_doc)

print("分割后的文本块：")
for idx,chunk in enumerate(chunk_list):
    print(f"片段{idx+1}\n{chunk}\n")
