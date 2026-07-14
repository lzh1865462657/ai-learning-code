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

## 异常处理

项目会记录向量库加载、文档解析、Ollama 调用和流式响应中的底层异常，并向调用方返回稳定的中文错误信息：

- 普通问答和文档入库失败时返回 `400`、`422` 或 `503`，进程不会退出。
- Ollama 未启动或模型不可用时，普通问答会返回 `503` 和明确的错误原因。
- 流式问答已开始响应后出现异常时，会发送 SSE `error` 事件，而不是让服务崩溃。

## Linux 部署

以下命令以 Ubuntu/Debian 服务器为例：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
git clone https://github.com/lzh1865462657/ai-learning-code.git
cd ai-learning-code/full_rag_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装 Ollama 并下载项目使用的模型：

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen:7b
ollama pull nomic-embed-text
```

后台启动服务并把日志写入文件：

```bash
mkdir -p logs
nohup .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > logs/rag-server.log 2>&1 &
tail -f logs/rag-server.log
```

查看和停止服务：

```bash
ps aux | grep "uvicorn main:app"
kill <进程号>
```

生产环境建议改用 systemd、反向代理和 HTTPS；`nohup` 适合学习、演示和基础服务器部署。

## 验收清单

1. 执行 `ollama list`，确认 `qwen:7b` 与 `nomic-embed-text` 已安装。
2. 调用 `/documents/index`，确认成功后生成本地 `chroma_db/` 目录。
3. 启动服务并访问 `http://127.0.0.1:8000/docs`。
4. 验证 `/knowledge_chat` 与 `/stream_chat` 能返回知识库答案。
5. 停止 Ollama 后再次调用接口，确认获得友好错误提示且 FastAPI 服务仍在运行。
