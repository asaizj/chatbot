from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import utils.evaluation as evaluation

# Se crea una cadena de conversacion utilizando conversaciones anteriores para mejorar la experiencia de usuario
def get_conversation_chain(vectorstore):
    # Se define una plantilla para los prompts con el fin de estructurar las preguntas y mejorar la precision del chatbot
    prompt_template = PromptTemplate(input_variables=["question", "context"],
    template="""
    Eres un asistente de seguros del hogar. Responde la siguiente pregunta de manera detallada y precisa.
    
    Contexto: {context}
    Pregunta: {question}
    Respuesta:
    """)
    # Se crea modelo de lenguaje basado en OpenAI con poca aleatoriedad en las respuesta (temperature)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0.1)
    # Se almacena y recupera conversaciones anteriores para mejorar las respuestas
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    retriever = vectorstore.as_retriever(search_type = "similarity", top_k = 5)
    # Se crea cadena de conversacion utilizando el modelo de lenguaje, el recuperador basado en vectores, y la memoria
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = retriever, memory = memory, combine_docs_chain_kwargs = {'prompt': prompt_template})
    # Se evalua el modelo
    evaluation.evaluation(llm)
    evaluation.evaluation_langsmith()
    return conversation_chain