from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DB_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL


def load_vector_store() -> Chroma:
    """Open the persisted Chroma collection used by this demo."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings,
    )


def add_documents(documents: Sequence[Document]) -> Chroma:
    """Store document chunks in the local Chroma database."""
    if not documents:
        raise ValueError("没有可写入向量库的文档内容")

    vector_store = load_vector_store()
    vector_store.add_documents(documents)
    return vector_store


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Return the most relevant source text for a user question."""
    documents = load_vector_store().similarity_search(question, k=top_k)
    return "\n\n".join(document.page_content for document in documents)
