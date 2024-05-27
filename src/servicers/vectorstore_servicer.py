import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

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