from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from uuid import UUID, uuid4
from typing import Literal


class ChunkMetadata(BaseModel):
    id: str
    chunk_index: int
    parent_document_id: str
    chunking_method: str
    start_index: int
    end_index: int
    source: str


class DocumentSplitter(ABC):
    add_start_index: bool = True

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def split(self, text: Document) -> list[Document]:
        pass

    def _add_metadata(
        self, docs: list[Document], document_uuid: UUID, chunking_method: str
    ) -> list[Document]:
        for i, doc in enumerate(docs):
            metadata = ChunkMetadata(
                id=str(uuid4()),
                chunk_index=i,
                parent_document_id=str(document_uuid),
                chunking_method=chunking_method,
                start_index=doc.metadata["start_index"],
                end_index=doc.metadata["start_index"] + len(doc.page_content),
                source=doc.metadata["source"],
            )
            doc.metadata = metadata.model_dump()
        return docs


class RecursiveSplitter(DocumentSplitter):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=self.add_start_index,
        )

    def split(self, text: Document) -> list[Document]:
        docs: list[Document] = self.splitter.create_documents(
            [text.page_content], metadatas=[text.metadata] if text.metadata else None
        )
        return self._add_metadata(docs, text.id, "RECURSIVE")


class SemanticSplitter(DocumentSplitter):
    def __init__(
        self,
        embedding_model,
        buffer_size: int = 1,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        breakpoint_threshold_amount: float | None = None,
        number_of_chunks: int | None = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: int | None = None,
    ) -> None:
        self.splitter = SemanticChunker(
            embeddings=embedding_model,
            buffer_size=buffer_size,
            add_start_index=self.add_start_index,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            breakpoint_threshold_type=breakpoint_threshold_type,
            number_of_chunks=number_of_chunks,
            sentence_split_regex=sentence_split_regex,
            min_chunk_size=min_chunk_size,
        )

    def split(self, text: Document) -> list[Document]:
        docs: list[Document] = self.splitter.create_documents(
            [text.page_content], metadatas=[text.metadata] if text.metadata else None
        )
        return self._add_metadata(docs, text.id, "SEMANTIC")
