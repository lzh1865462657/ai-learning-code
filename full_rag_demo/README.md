# Full RAG Demo

一个基于 FastAPI、Ollama、LangChain 和 Chroma 的本地 RAG（检索增强生成）学习项目。它支持将 TXT/PDF 文档写入本地向量库，并提供普通问答和流式问答接口。

## 功能

- TXT/PDF 文档加载、分块和向量化入库
- Chroma 本地持久化向量库检索
- Ollama 本地模型生成回答
- FastAPI 普通问答与 SSE 流式问答接口

## 环境要求

- Python 3.10+
- 已安装并运行 [Ollama](https://ollama.com/)
- 已下载模型：`qwen:7b` 和 `nomic-embed-text`

```bash
ollama pull qwen:7b
ollama pull nomic-embed-text
```

## 安装与运行

```bash
cd full_rag_demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

服务启动后访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

## 文档入库

命令行方式：

```bash
python document_loader.py path\to\document.pdf
```

或者调用接口：

```bash
curl -X POST http://127.0.0.1:8000/documents/index -H "Content-Type: application/json" -d "{\"file_path\": \"C:\\docs\\example.pdf\"}"
```

## 问答接口

普通问答：

```bash
curl -X POST http://127.0.0.1:8000/knowledge_chat -H "Content-Type: application/json" -d "{\"question\": \"RAG 是什么？\"}"
```

流式问答：

```bash
curl -N -X POST http://127.0.0.1:8000/stream_chat -H "Content-Type: application/json" -d "{\"question\": \"RAG 是什么？\"}"
```

## 项目结构

```text
full_rag_demo/
├── config.py           # 全局配置：地址、模型名称、向量库路径
├── document_loader.py  # PDF/TXT 文档加载与入库
├── vector_store.py     # 向量库加载与检索工具
├── llm_service.py      # 大模型调用、流式/非流式问答逻辑
├── main.py             # FastAPI 接口入口
├── README.md           # 项目说明
└── chroma_db/          # 运行时生成的向量库目录（不提交）
```

## 配置

在 `config.py` 中调整 Ollama 地址、模型名称、文本分块大小和向量库路径。`chroma_db/` 是运行时数据，已被 Git 忽略，不会上传到 GitHub。
