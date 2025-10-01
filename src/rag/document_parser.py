from langchain_community.document_loaders import PyPDFLoader, UnstructuredAPIFileLoader
from langchain_core.documents import Document
from pathlib import Path
from uuid import uuid4


UNSTRUCTURED_API_ENDPOINT = "http://localhost:8000/general/v0/general"


def load_document(path: Path, use_api: bool = False) -> list[Document]:
    suffix = path.suffix.lower()

    if use_api:
        loader = UnstructuredAPIFileLoader(file_path=str(path), url=UNSTRUCTURED_API_ENDPOINT, mode="elements")

    if suffix == ".pdf":
        loader = PyPDFLoader(path, mode="single")

    docs = loader.load()
    for doc in docs:
        doc.id = str(uuid4())
        doc.metadata["source"] = str(path)

    return docs
