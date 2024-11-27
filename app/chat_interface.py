import gradio as gr

def create_chat_interface(conversation_chain):
    def chat(message, history):
        result = conversation_chain.invoke({"question": message})
        return result["answer"]
    
    return gr.ChatInterface(chat).launch()