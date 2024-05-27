import fitz
import streamlit as st
from io import BytesIO

# Metodo de extracction de texto a partir de PDF con un mejor manejo de los caracteres especiales
def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        document = fitz.open(stream=BytesIO(pdf_files.read()), filetype = "pdf")
        #document = fitz.open(pdf)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    return text