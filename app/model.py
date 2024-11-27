from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI

from app.config import MODEL

def setup_conversational_chain(vector_store):
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 25})

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,callbacks=[StdOutCallbackHandler()]
    )
