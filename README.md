# RAG-Language Model for Insurance Knowledge Worker

A **Retrieval-Augmented Generation (RAG)**-based question-answering system designed for employees at Insurellm, an insurance technology company. This application ensures high accuracy and cost-effectiveness by utilizing a knowledge base of documents and a GPT-powered language model.

---
## Demo

https://github.com/user-attachments/assets/02f572cd-828d-4261-9cac-f505035b8500



## Features

- **Document Parsing & Knowledge Base Construction:**
  - Automatically loads and processes documents from the `knowledge-base` directory.
  - Splits documents into smaller, manageable chunks for efficient storage and retrieval.
- **Vector Store Creation:**
  - Uses **OpenAI embeddings** to generate vectorized representations of document chunks.
  - Stores vectors in a **Chroma** database for optimized retrieval.
- **Conversational Interface:**
  - Leverages a GPT-based language model (e.g., GPT-4) for conversational responses.
  - Supports a memory buffer for multi-turn conversations.

---

## Tech Stack

- **Language Models**: GPT (via OpenAI API)
- **RAG Framework**: LangChain
- **Vector Store**: Chroma
- **Frontend**: Gradio for interactive chat interface
- **Environment Management**: Python with `.env` configuration

---

## Directory Structure

```
├── app
│   ├── __init__.py
│   ├── config.py               # Application configuration (e.g., API keys)
│   ├── data_loader.py          # Logic for loading and splitting documents
│   ├── vector_store.py         # Vector store initialization and management
│   ├── chat_interface.py       # Gradio-based chat interface
│   ├── conversation_chain.py   # Retrieval-augmented chain and conversation memory
├── knowledge-base              # Directory containing knowledge base documents
├── vector_db                   # Persistent Chroma vector store
├── venv                        # Python virtual environment
├── .env                        # Environment variables (e.g., OpenAI API key)
├── .gitignore                  # Files to ignore in Git
├── requirements.txt            # Project dependencies
├── run.py                      # Entry point for the application
├── README.md                   # Project documentation
```

## Installation

1. Clone the Repository:
   ```
   git clone https://github.com/your-repo-name/rag-llm-insurance.git
   cd rag-llm-insurance
   ```
2. Set Up Python Environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Add Environment Variables: Create a .env file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Prepare Knowledge Base: Add your Markdown files to the knowledge-base folder. Ensure subfolders correspond to specific document types (e.g., policies, guidelines).
5. Run the Application:
   ```
   python run.py
   ```

## Usage

1. Open the Gradio interface in your browser.
2. Enter your query in the chatbox.
3. The system will retrieve the most relevant information from the knowledge base and generate a response using GPT.
