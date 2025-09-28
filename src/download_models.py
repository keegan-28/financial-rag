from sentence_transformers import SentenceTransformer
import os
from src.rag.utils import logger


def download_embedding_model(model_name: str, save_dir: str) -> None:
    """
    Download and save a SentenceTransformer embedding model from Hugging Face.
    """
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Downloading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Saving model to {save_dir}")
    model.save(save_dir)
    logger.info("Download complete.")


if __name__ == "__main__":
    download_embedding_model(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        save_dir="./models/all-MiniLM-L6-v2",
    )
