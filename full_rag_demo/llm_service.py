import json
from collections.abc import Generator

import requests

from config import LLM_MODEL, OLLAMA_API_URL, REQUEST_TIMEOUT_SECONDS
from vector_store import retrieve_context


def _build_prompt(question: str) -> str:
    context = retrieve_context(question)
    return f"""你是私有知识库问答助手，严格遵守以下规则：
1. 只能使用【参考知识库】中的内容回答问题；没有相关内容时回复“暂无对应知识库资料”。
2. 禁止编造知识点，回答应简洁、准确、易懂。

【参考知识库】
{context}

【用户提问】
{question}
"""


def answer_question(question: str) -> str:
    """Generate a complete RAG answer through the local Ollama API."""
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": LLM_MODEL, "prompt": _build_prompt(question), "stream": False},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"]


def stream_answer(question: str) -> Generator[str, None, None]:
    """Yield response tokens from Ollama as server-sent events."""
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": LLM_MODEL, "prompt": _build_prompt(question), "stream": True},
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        token = json.loads(line).get("response", "")
        if token:
            yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"

    yield "event: done\ndata: {}\n\n"
