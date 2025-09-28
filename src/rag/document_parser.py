from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(path, mode="single")

    return loader.load()
