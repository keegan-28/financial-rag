import dagster as dg
from src.rag.utils import logger
from pathlib import Path
from langchain_core.documents import Document
from src.rag.document_parser import load_document
from src.rag.document_chunking import RecursiveSplitter
from .resources import VectorStoreResource, DocumentStoreResource


@dg.asset
def datasets() -> list[Path]:
    base_path = Path("./data/dataset/financial_docs")
    dataset_paths = [item for item in base_path.iterdir() if item.is_file()]
    logger.info(f"Retrieved {len(dataset_paths)}")
    return dataset_paths


@dg.asset(deps=[datasets])
def documents(datasets: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for document_path in datasets:
        logger.info(f"Loading {document_path}")
        try:
            file_docs = load_document(document_path)
            documents.extend(file_docs)
        except Exception as e:
            logger.warning(f"Failed to process file {document_path}: {e}")
    logger.info(f"Loaded {len(documents)} from dataset.")
    return documents


@dg.asset(deps=[documents])
def chunked_documents(
    documents: list[Document], document_store_resource: DocumentStoreResource
) -> list[Document]:
    document_store = document_store_resource.get_client()
    logger.info(f"Chunking {len(documents)} documents.")
    recursive_splitter = RecursiveSplitter(1000, 100)
    chunked_docs: list[Document] = []

    for doc in documents:
        chunked_docs.extend(recursive_splitter.split(doc))

    document_store.write_documents(chunked_docs)
    logger.info(f"Created and stored {len(chunked_docs)} chunks.")
    return chunked_docs


@dg.asset(deps=[chunked_documents])
def embed(
    chunked_documents: list[Document], vector_store_resource: VectorStoreResource
) -> None:
    vector_store = vector_store_resource.get_client()
    logger.info(f"Embedding {len(chunked_documents)} documents.")
    vector_store.add_documents(chunked_documents)
    logger.info(f"Stored {len(chunked_documents)} documents.")
