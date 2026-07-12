"""A small, offline TF-IDF retriever that demonstrates the core RAG flow."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese characters without extra packages."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True, slots=True)
class Document:
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    score: float


class TfidfRetriever:
    """Index documents and rank them by cosine similarity to a query."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("知识库至少需要一篇文档")

        self._document_tokens = [tokenize(f"{doc.title} {doc.content}") for doc in self.documents]
        document_frequency = Counter(
            token for tokens in self._document_tokens for token in set(tokens)
        )
        count = len(self.documents)
        self._idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self._vectors = [self._vectorize(tokens) for tokens in self._document_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens) or 1
        return {
            token: (frequency / total) * self._idf.get(token, 0.0)
            for token, frequency in counts.items()
            if token in self._idf
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        common = left.keys() & right.keys()
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value**2 for value in left.values()))
        right_norm = math.sqrt(sum(value**2 for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("查询内容不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")

        query_vector = self._vectorize(tokenize(query))
        ranked = [
            SearchResult(document=document, score=round(self._cosine(query_vector, vector), 4))
            for document, vector in zip(self.documents, self._vectors, strict=True)
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return [result for result in ranked[:top_k] if result.score > 0]

    def answer(self, query: str, top_k: int = 3) -> dict[str, object]:
        """Return retrieved context as a deterministic, inspectable answer."""

        results = self.search(query, top_k=top_k)
        if not results:
            return {
                "answer": "知识库中没有找到相关内容，请换一种问法或补充知识文档。",
                "sources": [],
            }

        context = "\n".join(f"- {result.document.content}" for result in results)
        return {
            "answer": f"根据本地知识库：\n{context}",
            "sources": [
                {"title": result.document.title, "score": result.score} for result in results
            ],
        }


def load_documents(path: str | Path) -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Document(title=record["title"], content=record["content"]) for record in records]
