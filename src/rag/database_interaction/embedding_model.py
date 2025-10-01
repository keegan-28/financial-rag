import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


embedding_model_local = HuggingFaceEmbeddings(
    model_name="./models/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", max_retries=2)
