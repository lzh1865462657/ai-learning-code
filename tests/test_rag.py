from ai_learning.rag import Document, TfidfRetriever, tokenize


def test_tokenize_supports_chinese_and_english() -> None:
    tokens = tokenize("FastAPI 如何启动")

    assert "fastapi" in tokens
    assert "启" in tokens


def test_retriever_returns_relevant_document() -> None:
    retriever = TfidfRetriever(
        [
            Document("FastAPI", "使用 uvicorn 启动 API 服务"),
            Document("Pandas", "使用 groupby 聚合表格数据"),
        ]
    )

    result = retriever.search("uvicorn 启动", top_k=1)

    assert result[0].document.title == "FastAPI"
    assert result[0].score > 0


def test_retriever_handles_unknown_query() -> None:
    retriever = TfidfRetriever([Document("Python", "列表和字典是常用数据结构")])

    response = retriever.answer("量子芯片")

    assert response["sources"] == []
