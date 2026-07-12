# ai-learning-code

AI 专业实习学习代码，覆盖 Python 基础、NumPy/Pandas 数据分析、FastAPI 接口和一个不依赖外部大模型服务的轻量 RAG 项目。

## 项目结构

```text
ai-learning-code/
|-- data/                     # 示例销售数据和 RAG 知识库
|-- examples/                 # 各模块的可直接运行示例
|-- src/ai_learning/
|   |-- python_basics.py      # 数据类、集合处理、成绩统计
|   |-- data_analysis.py      # Pandas 清洗、聚合与 NumPy 指标计算
|   |-- rag.py                # TF-IDF 检索和答案生成
|   `-- api.py                # FastAPI 健康检查、问答和批量问答接口
|-- tests/                    # pytest 自动化测试
`-- pyproject.toml            # 依赖和工具配置
```

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

## 运行示例

```bash
python examples/01_python_basics.py
python examples/02_data_analysis.py
python examples/03_rag_demo.py
```

启动 API：

```bash
uvicorn ai_learning.api:app --reload
```

浏览器访问 `http://127.0.0.1:8000/docs`，可直接调试以下接口：

- `GET /health`：服务健康检查
- `POST /chat`：从本地知识库检索并回答问题
- `POST /batch-chat`：批量问答

单次问答示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"FastAPI 如何启动？","top_k":2}'
```

## 学习路线

1. 阅读并运行 `python_basics.py`，熟悉类型标注、数据类、列表推导式和异常处理。
2. 修改 `sales.csv` 后运行数据分析示例，观察清洗和聚合结果。
3. 阅读 `rag.py`，理解分词、TF-IDF、余弦相似度和检索增强生成的基本流程。
4. 通过 FastAPI 的 `/docs` 页面调用问答服务，并尝试扩充 `knowledge.json`。
