from app.config import OPENAI_API_KEY
from app.data_loader import load_documents, split_documents
from app.vector_store import initialize_vector_store
from app.model import setup_conversational_chain
from app.chat_interface import create_chat_interface

def main():
    if not OPENAI_API_KEY:
        raise EnvironmentError("OpenAI API key not set in .env file.")

    # Load and process documents
    documents = load_documents()
    chunks = split_documents(documents)

    # Initialize vector store and conversational chain
    vector_store = initialize_vector_store(chunks)
    conversation_chain = setup_conversational_chain(vector_store)

    # Launch chat interface
    chat_interface = create_chat_interface(conversation_chain)
    chat_interface.launch()

if __name__ == "__main__":
    main()