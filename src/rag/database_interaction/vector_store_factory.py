from langchain_community.vectorstores import Chroma
from .embedding_model import embedding_model_local, embedding_model_api


class VectorStoreFactory:
    @classmethod
    def get_vector_store(
        self,
        collection_name: str = "FinancialCollection",
        persistent_directory: str = "./database/chroma",
        local: bool = True,
    ) -> Chroma:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model_local if local else embedding_model_api,
            persist_directory=persistent_directory,
        )

        return vector_store
