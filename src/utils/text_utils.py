from langchain_text_splitters import RecursiveCharacterTextSplitter

# Recibe una cadena de texto (el texto de todos los PDFs leidos previamente) y la divide en fragmentos mas pequeños
def get_text_chunks(text):
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
    chunks = text_splitter.split_text(text)
    return chunks