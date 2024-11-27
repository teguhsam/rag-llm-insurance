import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import DB_NAME

def initialize_vector_store(documents, persist_directory=DB_NAME):
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=persist_directory, embedding_function=embeddings).delete_collection()

    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    return vector_store