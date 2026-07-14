from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, configure_logging
from vector_store import VectorStoreError, add_documents

logger = logging.getLogger(__name__)


class DocumentProcessingError(RuntimeError):
    """Raised when a source file cannot be loaded or split."""


def _load_source_document(document_path: Path) -> list[Document]:
    """Load a supported source file, falling back to GBK for legacy TXT files."""
    suffix = document_path.suffix.lower()
    if suffix == ".txt":
        try:
            return TextLoader(str(document_path), encoding="utf-8").load()
        except UnicodeDecodeError:
            logger.info("Retrying text document with GBK encoding: %s", document_path)
            return TextLoader(str(document_path), encoding="gbk").load()
    if suffix == ".pdf":
        return PyPDFLoader(str(document_path)).load()
    raise ValueError("仅支持 .txt 和 .pdf 文件")


def load_documents(file_path: str | Path) -> list[Document]:
    """Load and split one TXT or PDF file into retrieval chunks."""
    document_path = Path(file_path).resolve()

    if not document_path.is_file():
        raise FileNotFoundError(f"找不到文档：{document_path}")

    try:
        docs = _load_source_document(document_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise DocumentProcessingError("文档没有可入库的文本内容")
        return chunks
    except (DocumentProcessingError, ValueError):
        raise
    except Exception as error:
        logger.exception("Failed to load document: %s", document_path)
        raise DocumentProcessingError(f"文档解析失败：{document_path.name}") from error


def index_document(file_path: str | Path) -> int:
    """Load a TXT/PDF file and add its chunks to the local vector store."""
    document_path = Path(file_path).resolve()
    try:
        chunks = load_documents(document_path)
        add_documents(chunks)
        logger.info("Indexed document %s with %s chunks", document_path, len(chunks))
        return len(chunks)
    except (FileNotFoundError, ValueError, DocumentProcessingError, VectorStoreError):
        raise
    except Exception as error:
        logger.exception("Failed to index document: %s", document_path)
        raise DocumentProcessingError(f"文档入库失败：{document_path.name}") from error


if __name__ == "__main__":
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser(description="将 TXT 或 PDF 文档写入 Chroma 向量库")
    parser.add_argument("file_path", help="要入库的 TXT 或 PDF 文件路径")
    args = parser.parse_args()
    try:
        chunk_count = index_document(args.file_path)
        print(f"文档入库完成，共 {chunk_count} 个分块")
    except (FileNotFoundError, ValueError, DocumentProcessingError, VectorStoreError) as error:
        parser.exit(1, f"文档入库失败：{error}\n")
