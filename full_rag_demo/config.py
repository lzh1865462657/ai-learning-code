from pathlib import Path
import logging


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen:7b"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
REQUEST_TIMEOUT_SECONDS = 120


def configure_logging() -> None:
    """Configure console logging for local development and nohup deployment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
