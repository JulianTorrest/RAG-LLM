import streamlit as st
import requests
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Lista para almacenar los embeddings y los textos
documentos = []
embeddings = []

# URL del archivo PDF en GitHub
GITHUB_PDF_URL = "https://raw.githubusercontent.com/JulianTorrest/RAG-LLM/main/1210-Insurance-2030-The-impact-of-AI-on-the-future-of-insurance-_-McKinsey-Company.pdf"

# Inicializar el traductor
translator = Translator()

def descargar_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("documento.pdf", "wb") as f:
            f.write(response.content)
        return "documento.pdf"
    return None

def extraer_texto(pdf_path):
    doc = fitz.open(pdf_path)
    texto = "".join([page.get_text("text") for page in doc])
    return texto

def indexar_texto(texto):
    global documentos, embeddings
    chunks = texto.split(". ")  # Dividir en oraciones
    for chunk in chunks:
        embedding = embedding_model.encode(chunk)
        documentos.append(chunk)
        embeddings.append(embedding)

def buscar_similaridades(query):
    query_embedding = embedding_model.encode(query)
    similitudes = cosine_similarity([query_embedding], embeddings)
    idx_similares = np.argsort(similitudes[0])[::-1][:3]  # Los tres más similares
    return "\n".join([documentos[i] for i in idx_similares])

# Función para traducir el texto
def traducir_a_ingles(texto):
    return translator.translate(texto, src='es', dest='en').text

# Función para traducir la respuesta al español
def traducir_a_espanol(texto):
    return translator.translate(texto, src='en', dest='es').text

# Interfaz en Streamlit
st.title("RAG con Streamlit y GitHub")

# Descargar y procesar el PDF automáticamente al cargar la app
st.write("Procesando PDF...")

pdf_path = descargar_pdf(GITHUB_PDF_URL)
if pdf_path:
    texto = extraer_texto(pdf_path)
    indexar_texto(texto)
    st.success("PDF procesado e indexado correctamente.")

pregunta = st.text_input("Haz una pregunta sobre el documento en español")

if st.button("Buscar respuesta") and pregunta:
    # Traducir la pregunta a inglés
    pregunta_ingles = traducir_a_ingles(pregunta)
    respuesta_ingles = buscar_similaridades(pregunta_ingles)
    # Traducir la respuesta al español
    respuesta_espanol = traducir_a_espanol(respuesta_ingles)
    st.write(respuesta_espanol)
