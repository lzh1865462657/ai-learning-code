from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from vector_store import add_documents


def load_documents(file_path: str | Path) -> list[Document]:
    """Load and split one UTF-8 TXT or PDF file into retrieval chunks."""
    document_path = Path(file_path).resolve()
    suffix = document_path.suffix.lower()

    if not document_path.is_file():
        raise FileNotFoundError(f"找不到文档：{document_path}")

    if suffix == ".txt":
        loader = TextLoader(str(document_path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(document_path))
    else:
        raise ValueError("仅支持 .txt 和 .pdf 文件")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def index_document(file_path: str | Path):
    """Load a TXT/PDF file and add its chunks to the local vector store."""
    document_path = Path(file_path).resolve()
    chunks = load_documents(document_path)
    vector_store = add_documents(chunks)
    print(f"{document_path.name} 文档入库完成，共 {len(chunks)} 个分块")
    return vector_store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 TXT 或 PDF 文档写入 Chroma 向量库")
    parser.add_argument("file_path", help="要入库的 TXT 或 PDF 文件路径")
    args = parser.parse_args()
    index_document(args.file_path)
