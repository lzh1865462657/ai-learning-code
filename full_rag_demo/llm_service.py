import json
from collections.abc import Generator
import logging

import requests

from config import LLM_MODEL, OLLAMA_API_URL, REQUEST_TIMEOUT_SECONDS
from vector_store import VectorStoreError, retrieve_context

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when a RAG answer cannot be generated."""


def _build_prompt(question: str) -> str:
    try:
        context = retrieve_context(question)
    except (ValueError, VectorStoreError) as error:
        raise LLMServiceError(str(error)) from error
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
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": LLM_MODEL, "prompt": _build_prompt(question), "stream": False},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json().get("response")
        if not answer:
            raise LLMServiceError("Ollama 没有返回有效回答")
        return answer
    except LLMServiceError:
        raise
    except requests.exceptions.ConnectionError as error:
        logger.warning("Unable to connect to Ollama at %s", OLLAMA_API_URL)
        raise LLMServiceError("无法连接 Ollama，请确认 Ollama 程序和模型已启动") from error
    except requests.RequestException as error:
        logger.exception("Ollama request failed")
        raise LLMServiceError("Ollama 调用失败，请稍后重试") from error
    except (json.JSONDecodeError, ValueError, KeyError) as error:
        logger.exception("Invalid Ollama response")
        raise LLMServiceError("Ollama 返回了无法解析的结果") from error


def stream_answer(question: str) -> Generator[str, None, None]:
    """Yield response tokens from Ollama as server-sent events."""
    try:
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
    except LLMServiceError as error:
        logger.warning("Unable to start RAG stream: %s", error)
        yield _error_event(str(error))
        return
    except requests.exceptions.ConnectionError:
        logger.warning("Unable to connect to Ollama at %s", OLLAMA_API_URL)
        yield _error_event("无法连接 Ollama，请确认 Ollama 程序和模型已启动")
        return
    except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
        logger.exception("RAG stream failed")
        yield _error_event("流式问答服务异常，请稍后重试")
        return

    yield "event: done\ndata: {}\n\n"


def _error_event(message: str) -> str:
    return f"event: error\ndata: {json.dumps({'message': message}, ensure_ascii=False)}\n\n"
