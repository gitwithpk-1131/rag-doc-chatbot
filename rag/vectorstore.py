import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vectorstore(chunks):
    model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=model)
    return Chroma.from_documents(chunks, embeddings)