import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

def main():
    # Cargamos las variables de entorno (KEYs)
    #load_dotenv()
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
        openai_api_key = st.text_input("Introduzca API Key de OpenAI", key = "chatbot_api_key", type = "password")
        process = st.button("Procesar")
    # Se comprueba si se ha introducido el API Key. Si no, finaliza la ejecucion.    
    if process:
        #if not openai_api_key:
        #    st.info("Por favor, introduce tu API Key de OpenAI para continuar.")
        #    st.stop()
        #else:
        with st.spinner("Procesando..."):
            # Se cargan los archivos
            files_text = get_pdf_text(pdf_files)
            # Se dividen en chunks
            text_chunks = get_text_chunks(files_text)
            # Se almacenan en la base de datos vectorial
            vetorstore = get_vectorstore(text_chunks, openai_api_key)
        # Se genera una cadena de conversacion
        st.session_state.conversation = get_conversation_chain(vetorstore, openai_api_key) 

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("¿Tienes dudas con tu póliza de seguro de hogar? ¡Pregúntame!")
        if user_question:
            handle_userinput(user_question)

# Recibe una lista de archivos PDF y devuelve el texto extraido de estos archivos en una unica cadena
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        # Se itera sobre cada pagina del PDF y se extrae el texto
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Recibe una cadena de texto (el texto de todos los PDFs leidos previamente) y la divide en fragmentos mas pequeños
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        # Se utiliza como separador el salto de linea
        separator = "\n",
        # Se establece el tamaño maximo de cada fragmento de texto = 1000 caracteres
        chunk_size = 1000,
        # Se define la cantidad de superposicion entre fragmentos adyacentes. Util para asegurar que no se pierda informacion en los limites de los fragmentos
        chunk_overlap = 200,
        length_function = len
    )
    # Probar con RecursiveCharacterTextSplitter
    chunks = text_splitter.split_text(text)
    return chunks

# Recibe una lista de chunks y genera un almacenamiento de vectores utilizando embeddings
def get_vectorstore(text_chunks, openai_api_key):
    # Embeddings para representar el texto en forma de vectores numericos
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    # Se crear un vector store a partir de los chunks (busqueda de vectores)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

# Se crea una cadena de conversacion utilizando conversaciones anteriores para mejorar la experiencia de usuario
def get_conversation_chain(vectorstore, openai_api_key):
    # Se crea modelo de lenguaje basado en OpenAI con poca aleatoriedad en las respuesta (temperature)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0, openai_api_key = openai_api_key)
    # Se almacena y recupera conversaciones anteriores para mejorar las respuestas
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    # Se crea cadena de conversacion utilizando el modelo de lenguaje, el recuperador basado en vectores, y la memoria
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(search_type = "similarity"), memory = memory)
    return conversation_chain

# Manejar la entrada de datos del usuario
def handle_userinput(user_question):
    # Se pasa la pregunta del usuario a la cadena de conversacion
    response = st.session_state.conversation({'question': user_question})
    # Se actualiza el historial de chat almacenado en la sesion con la respuesta generada por el chatbot
    st.session_state.chat_history = response['chat_history']
    # Se itera sobre cada mensaje en el historial de chat. Si el indice del mensaje es par, se trata de un mensaje del usuario y se muestra por pantalla
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(message.content, is_user = True, key = str(i))
        # Si el indice es impar, asume que se trata de una respuesta del chatbot
        else:
            st.write(message.content, key = str(i))

if __name__ == '__main__':
    main()
