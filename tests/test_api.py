from fastapi.testclient import TestClient

from ai_learning.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["document_count"] >= 1


def test_chat() -> None:
    response = client.post("/chat", json={"question": "FastAPI 如何启动？", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "FastAPI 如何启动？"
    assert body["sources"][0]["title"] == "FastAPI 启动方式"


def test_batch_chat() -> None:
    response = client.post(
        "/batch-chat",
        json={"questions": ["什么是 RAG？", "Pandas 如何读取 CSV？"]},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 2
