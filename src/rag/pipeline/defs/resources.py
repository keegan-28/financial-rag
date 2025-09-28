from dagster import ConfigurableResource, EnvVar
from src.rag.database_interaction.document_database import DocumentStore
from src.rag.database_interaction.vector_store_factory import VectorStoreFactory
from langchain_community.vectorstores import Chroma


class DocumentStoreResource(ConfigurableResource):
    database_path: str

    def get_client(self) -> DocumentStore:
        return DocumentStore(database_path=self.database_path)


class VectorStoreResource(ConfigurableResource):
    collection_name: str
    database_path: str
    api_key: str

    def get_client(self) -> Chroma:
        return VectorStoreFactory.get_vector_store(
            collection_name=self.collection_name,
            persistent_directory=self.database_path,
        )


document_store_resource = DocumentStoreResource(
    database_path=EnvVar("DOCUMENT_STORE_PATH"),
)

vector_store_resource = VectorStoreResource(
    collection_name=EnvVar("VECTOR_STORE_COLLECTION"),
    database_path=EnvVar("VECTOR_STORE_PATH"),
    api_key=EnvVar("GEMINI_API_KEY"),
)
