from collections.abc import Sequence
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DB_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the local vector store cannot be used."""


def load_vector_store() -> Chroma:
    """Open the persisted Chroma collection used by this demo."""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings,
        )
    except Exception as error:
        logger.exception("Failed to load the vector store")
        raise VectorStoreError("向量库加载失败，请确认 Chroma 数据目录和嵌入模型可用") from error


def add_documents(documents: Sequence[Document]) -> Chroma:
    """Store document chunks in the local Chroma database."""
    if not documents:
        raise ValueError("没有可写入向量库的文档内容")

    try:
        vector_store = load_vector_store()
        vector_store.add_documents(documents)
        return vector_store
    except VectorStoreError:
        raise
    except Exception as error:
        logger.exception("Failed to add documents to the vector store")
        raise VectorStoreError("文档写入向量库失败，请确认 Ollama 嵌入模型已启动") from error


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Return the most relevant source text for a user question."""
    if not question.strip():
        raise ValueError("问题不能为空")

    try:
        documents = load_vector_store().similarity_search(question, k=top_k)
        return "\n\n".join(document.page_content for document in documents)
    except VectorStoreError:
        raise
    except Exception as error:
        logger.exception("Failed to retrieve related documents")
        raise VectorStoreError("知识库检索失败，请确认嵌入模型和向量库可用") from error
