from .vector_store_factory import VectorStoreFactory
import os


COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION")
PERSISTENT_PATH = os.getenv("VECTOR_STORE_PATH")


vector_store = VectorStoreFactory().get_vector_store(
    collection_name=COLLECTION_NAME, persistent_directory=PERSISTENT_PATH
)
