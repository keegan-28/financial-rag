from sqlmodel import SQLModel, Field, create_engine, Session, select
from langchain_core.documents import Document
from src.rag.document_chunking import ChunkMetadata


BASE_DB_URI = "sqlite:///"


class ChildDocuments(ChunkMetadata, SQLModel, table=True):
    id: str = Field(primary_key=True)
    text: str


class DocumentStore:
    def __init__(self, database_path: str):
        self.db_uri = f"{BASE_DB_URI}{database_path}"
        self.engine = create_engine(self.db_uri)
        SQLModel.metadata.create_all(self.engine)

    def write_documents(self, docs: list[Document]) -> None:
        inserts = []
        for i, doc in enumerate(docs):
            row = ChildDocuments(
                id=doc.metadata.get("id"),
                text=doc.page_content,
                chunk_index=doc.metadata.get("chunk_index", i),
                parent_document_id=doc.metadata.get("parent_document_id"),
                chunking_method=doc.metadata.get("chunking_method"),
                start_index=int(doc.metadata.get("start_index")),
                end_index=int(doc.metadata.get("end_index")),
                source=doc.metadata.get("source"),
            )
            inserts.append(row)

        with Session(self.engine) as session:
            for row in inserts:
                session.merge(row)
            session.commit()

    def read_documents(self, chunk_ids: list[str]) -> list[Document]:
        documents = []
        with Session(self.engine) as session:
            statement = select(ChildDocuments).where(ChildDocuments.id.in_(chunk_ids))
            results = session.exec(statement).all()
            for row in results:
                doc = Document(
                    page_content=row.text,
                    metadata={
                        "id": row.id,
                        "chunk_index": row.chunk_index,
                        "chunking_method": row.chunking_method,
                        "parent_document_id": row.parent_document_id,
                        "start_index": row.start_index,
                        "end_index": row.end_index,
                        "source": row.source,
                    },
                )
                documents.append(doc)
        return documents
