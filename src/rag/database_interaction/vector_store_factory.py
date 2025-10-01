from langchain_chroma import Chroma
from .embedding_model import embedding_model_local


class VectorStoreFactory:
    @classmethod
    def get_vector_store(
        self,
        collection_name: str = "FinancialCollection",
        persistent_directory: str = "./database/chroma",
    ) -> Chroma:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model_local,
            persist_directory=persistent_directory,
        )

        return vector_store
