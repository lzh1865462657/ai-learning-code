from pathlib import Path
from pprint import pprint

from ai_learning.rag import TfidfRetriever, load_documents

project_root = Path(__file__).resolve().parents[1]
retriever = TfidfRetriever(load_documents(project_root / "data" / "knowledge.json"))
pprint(retriever.answer("FastAPI 服务怎么启动？", top_k=2), sort_dicts=False)
