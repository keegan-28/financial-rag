from langchain.schema import Document
from pydantic import BaseModel
from typing import List
from src.rag.database_interaction.vector_store import (
    document_store,
)  # your SQLite wrapper
from src.rag.database_interaction.document_database import ParentDocuments
from collections import Counter


class CitedSentences(BaseModel):
    answer: str
    citations: List[str]


def format_docs_with_id(docs: list[Document]) -> str:
    """
    Formats retrieved docs into a context string with IDs for citation.
    Each doc should carry metadata (source, page, etc.)
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted_chunks.append(
            f"[Doc {i} | Source: {source}, Page: {page}]\n{doc.page_content}\n"
        )
    return "\n\n".join(formatted_chunks)


def format_cited_answer(output: CitedSentences) -> str:
    """
    Formats the LLM structured output into a final answer string with citations.
    """
    answer = output.answer
    citations = output.citations

    if citations:
        citations_str = " | ".join(citations)
        return f"{answer}\n\nCitations: {citations_str}"
    return answer


def _fetch_parent_docs(docs: list[Document]) -> dict[str, list[Document]]:
    parent_ids = list({doc.metadata["parent_document_id"] for doc in docs})
    parent_docs = document_store.read_documents(parent_ids, ParentDocuments)
    return {"docs": parent_docs}


def fetch_parent_docs_wrapper(d: list[Document]):
    # Count child hits per parent
    parent_ids = [doc.metadata["parent_document_id"] for doc in d["docs"]]
    parent_count = Counter(parent_ids)

    # Fetch parent docs (your existing function)
    parent_docs = _fetch_parent_docs(d["docs"])["docs"]

    # Attach hit count to metadata for later use
    for pd in parent_docs:
        pid = pd.metadata["id"]
        pd.metadata["hit_count"] = parent_count.get(pid, 1)

    return {
        "docs": parent_docs,  # parent docs only
        "query": d["query"],
        "original_query": d["original_query"],
    }


def format_docs_with_weight(docs: list[Document]) -> str:
    """
    Format parent docs with weight (hit count).
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        weight = doc.metadata.get("hit_count", 1)
        formatted_chunks.append(
            f"[Doc {i} | Source: {source}, Weight: {weight}]\n{doc.page_content}\n"
        )
    return "\n\n".join(formatted_chunks)


def weight_chunks_by_parent(
    docs: list[Document], alpha: float = 0.7, beta: float = 0.3
) -> list[dict]:
    # Count frequency of parent_ids
    parent_counts = Counter(
        doc.metadata.get("parent_document_id")
        for doc in docs
        if "parent_document_id" in doc.metadata
    )
    max_count = max(parent_counts.values()) if parent_counts else 1

    reranked: list[Document] = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_document_id")
        sim_score = doc.metadata.get("score", 1.0)
        parent_boost = parent_counts.get(parent_id, 0) / max_count

        weighted_score = alpha * sim_score + beta * parent_boost
        doc.metadata["weighted_score"] = weighted_score
        reranked.append(doc)

    # Sort by new score, highest first
    reranked.sort(key=lambda d: d.metadata["weighted_score"], reverse=True)
    return reranked
