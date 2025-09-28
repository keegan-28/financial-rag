import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_api = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)

embedding_model_local = HuggingFaceEmbeddings(
    model_name="./models/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
