
from dotenv import load_dotenv
import os
import sys
from langchain.callbacks.tracers import LangChainTracer
import streamlit as st
import utils.text_utils as utils_text
import utils.pdf_utils as utils_pdf
import servicers.vectorstore_servicer as vectorstore
import servicers.conversation_servicer as conversation
import handlers.user_handler as handler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    # Cargamos las variables de entorno
    load_dotenv()
    # Obtener los valores relacionados con Langsmith para trackear la app
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
    langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    langchain_project = os.getenv("LANGCHAIN_PROJECT")
    # Inicializamos las trazas para monitorizacion
    tracer = LangChainTracer(project_name = "default")

    st.set_page_config(page_title = "My Home insurance chatbot")
    st.header("Home insurance chatbot developed by atmira")

    # Inicializacion variables estado sesion: controlar flujo chatbot, almacenar estado actual conversacion e historial del chat
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        st.subheader("Tus documentos")
        pdf_files = st.file_uploader("Adjunta tu póliza en PDF.", type = ['pdf'], accept_multiple_files = True)
        process = st.button("Procesar")
    
    if process and pdf_files is not None:
        with st.spinner("Procesando..."):
            for pdf_file in pdf_files:
                try:
                    # Se cargan los archivos
                    #files_text = get_pdf_text(pdf_files)
                    files_text = utils_pdf.extract_text_from_pdf(pdf_file)
                    # Se dividen en chunks
                    text_chunks = utils_text.get_text_chunks(files_text)
                    # Se almacenan en la base de datos vectorial
                    vetorstore = vectorstore.get_vectorstore(text_chunks)
                except Exception as e:
                    st.error(f"Error al procesar el archivo: {e}")

        # Se genera una cadena de conversacion
        st.session_state.conversation = conversation.get_conversation_chain(vetorstore) 

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("¿Tienes dudas con tu póliza de seguro de hogar? ¡Pregúntame!")
        if user_question:
            handler.handle_userinput(user_question)

# Main
if __name__ == '__main__':
    main()