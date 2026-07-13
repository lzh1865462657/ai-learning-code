from langchain_ollama import OllamaEmbeddings
#调用本地模型
embedding=OllamaEmbeddings(model="nomic-embed-text")
#两端语义相近文字，一段无关文字测试相似度
text1="RAG用来做大模型私有知识库问答"
text2="检索增强生成可以读取本地文档回答问题"
text3="FastAPI适用于编写后端HTTP接口服务"
#shengchengxiangliang
vec1=embedding.embed_query(text1)
vec2=embedding.embed_query(text2)
vec3=embedding.embed_query(text3)
#打印向量长度
print(f"文本向量长度:{len(vec1)}")
print(f"与意向金文本向量距离更近，无关文本车距更大")
