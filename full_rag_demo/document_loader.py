from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"


def load_file_to_vector(file_path: str):
    document_path = Path(file_path).resolve()
    suffix = document_path.suffix.lower()

    if suffix == ".txt":
        loader = TextLoader(str(document_path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(document_path))
    else:
        raise ValueError("仅支持 .txt 和 .pdf 文件")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=str(DB_PATH),
    )
    print(f"{document_path.name} 文档入库完成，向量库位于 {DB_PATH}")
    return vector_store


if __name__ == "__main__":
    load_file_to_vector(BASE_DIR / "text.txt")
