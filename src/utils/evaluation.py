from langsmith import Client
from langchain.evaluation import load_evaluator, EvaluatorType

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