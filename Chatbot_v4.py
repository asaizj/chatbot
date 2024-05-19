import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.evaluation import load_evaluator, EvaluatorType
from dotenv import load_dotenv
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
import fitz  # PyMuPDF


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
        #openai_api_key = st.text_input("Introduzca API Key de OpenAI", key = "chatbot_api_key", type = "password")
        process = st.button("Procesar")
    
    if process:
        with st.spinner("Procesando..."):
            # Se cargan los archivos
            #files_text = get_pdf_text(pdf_files)
            files_text = extract_text_from_pdf(pdf_files)
            # Se dividen en chunks
            text_chunks = get_text_chunks(files_text)
            # Se almacenan en la base de datos vectorial
            vetorstore = get_vectorstore(text_chunks)

        # Se genera una cadena de conversacion
        st.session_state.conversation = get_conversation_chain(vetorstore) 

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

# Metodo de extracction de texto a partir de PDF con un mejor manejo de los caracteres especiales
def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        document = fitz.open(pdf)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    return text

# Recibe una cadena de texto (el texto de todos los PDFs leidos previamente) y la divide en fragmentos mas pequeños
def get_text_chunks(text):
    # Probar con RecursiveCharacterTextSplitter
    # text_splitter = CharacterTextSplitter(
    text_splitter = RecursiveCharacterTextSplitter(
        # Se utiliza como separador el salto de linea
        #separator = "\n",
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
def get_vectorstore(text_chunks):
    # Obtener el valor de la variable de entorno OPENAI_API_KEY
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Embeddings para representar el texto en forma de vectores numericos
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    # Se crear un vector store a partir de los chunks (busqueda de vectores)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    # Se guarda en local el vector store utilizado
    # vectorstore.save_local("faiss_index")
    return vectorstore

# Se crea una cadena de conversacion utilizando conversaciones anteriores para mejorar la experiencia de usuario
def get_conversation_chain(vectorstore):
    # Se genera un prompt inicial para que el chatbot lo utilice como contexto durante la conversacion 
    prompt_template = """ Eres un agente conversacional en un contexto de seguros del hogar. Responde a las preguntas de la manera más detallada posible teniendo en cuenta la información contenida en los PDF.
    Aségurate de proveer todos los detalles que aparezcan en los documentos según la pregunta realizada por el usuario. Si la respuesta a la pregunta no aparece en el documento, no respondas con información errónea.
    
    context: \n{context}?\n
    Question: \n{question} \n

    Answer:
    
    """
    # Se crea modelo de lenguaje basado en OpenAI con poca aleatoriedad en las respuesta (temperature)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0)
    # chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    # Se almacena y recupera conversaciones anteriores para mejorar las respuestas
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    # Se crea cadena de conversacion utilizando el modelo de lenguaje, el recuperador basado en vectores, y la memoria
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(search_type = "similarity"), memory = memory)
    # Se evalua el modelo
    evaluation(llm)
    evaluation_langsmith()
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
            #st.write(user_template.replace("{{MSG}}", message.content, is_user = True, key = str(i)), unsafe_allow_html = True)
            st.write(message.content, is_user = True, key = str(i))
        # Si el indice es impar, asume que se trata de una respuesta del chatbot
        else:
            #st.write(bot_template.replace("{{MSG}}", message.content, key = str(i)), unsafe_allow_html = True)
            st.write(message.content, key = str(i))

def evaluation(llm):
    # Definir posibles preguntas relacionadas con el seguro de hogar
    preguntas = [
        "¿Cuál es el precio de la póliza?",
        "¿Cuál es el importe de la póliza?",
        "¿Cuánto vale la póliza?",
        "¿Qué precio tiene la póliza?",
        "¿Qué precio tiene?",
        "¿Cuánto vale?"
    ]
    # Definir las respuestas correctas para cada pregunta
    respuestas_correctas = [
        "El precio de la póliza es 97,60€.",
        "El importe de la póliza es 97,60€.",
        "La póliza vale 97,60€.",
        "El precio de la póliza es 97,60€.",
        "El precio de la póliza es 97,60€.",
        "La póliza vale 97,60€."
    ]
    # Definir las respuestas del modelo para cada pregunta
    respuestas_modelo = [
        "El precio de la póliza es 97,60€.",
        "El importe de la póliza es 97,60€.",
        "La póliza vale 97,60€.",
        "El precio de la póliza es 97,60€.",
        "El precio de la póliza es 97,60€.",
        "La póliza vale 97,60€."
    ]
    evaluator = load_evaluator(EvaluatorType.LABELED_CRITERIA, llm = llm, criteria = "correctness") #correctness, relevance, coherence, conciseness
    # Evaluar la precisión y utilidad de las respuestas del modelo
    eval_result = evaluator.evaluate_strings(prediction = respuestas_modelo, reference = respuestas_correctas, input = preguntas)
    print(eval_result)

def evaluation_langsmith():
    # Se crea instancia del cliente para interactuar con la API de LangSmith
    client = Client()
    # Se definen preguntas relacionadas con el importe de la poliza
    input = [
        "¿Cuál es el precio de la póliza?",
        "¿Cuál es el importe de la póliza?",
        "¿Cuánto vale la póliza?",
        "¿Qué precio tiene la póliza?",
        "¿Qué precio tiene?",
        "¿Cuánto vale?"
    ]
    # Se define el nombre del dataset
    #dataset_name = "Importe poliza dataset"
    # Se crea el dataset en LangSmith
    #dataset = client.create_dataset(dataset_name = dataset_name, description = "Preguntas sobre el importe de la póliza.")
    # Para cada pregunta, se crea un ejemplo en el conjunto de datos en LangSmith
    #for input_prompt in input:
    #    client.create_example(inputs = {"question": input_prompt}, outputs = None, dataset_id = dataset.id)

# Main
if __name__ == '__main__':
    main()