import streamlit as st
import requests
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Lista para almacenar los embeddings y los textos
documentos = []
embeddings = []

# URL del archivo PDF en GitHub
GITHUB_PDF_URL = "https://raw.githubusercontent.com/JulianTorrest/RAG-LLM/main/1210-Insurance-2030-The-impact-of-AI-on-the-future-of-insurance-_-McKinsey-Company.pdf"

# Inicializar el traductor
translator = Translator()

# Función para descargar el PDF desde la URL
def descargar_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("documento.pdf", "wb") as f:
            f.write(response.content)
        return "documento.pdf"
    return None

# Función para extraer texto del PDF
def extraer_texto(pdf_path):
    doc = fitz.open(pdf_path)
    texto = "".join([page.get_text("text") for page in doc])
    return texto

# Función para indexar el texto en fragmentos (por oraciones)
def indexar_texto(texto):
    global documentos, embeddings
    chunks = texto.split(". ")  # Dividir en oraciones
    for chunk in chunks:
        embedding = embedding_model.encode(chunk)
        documentos.append(chunk)
        embeddings.append(embedding)

# Función para buscar similitudes de la pregunta con el contenido del documento
def buscar_similaridades(query):
    query_embedding = embedding_model.encode(query)
    similitudes = cosine_similarity([query_embedding], embeddings)
    idx_similares = np.argsort(similitudes[0])[::-1][:3]  # Los tres más similares
    return "\n".join([documentos[i] for i in idx_similares]), similitudes[0][idx_similares[0]]

# Función para traducir el texto a inglés
def traducir_a_ingles(texto):
    return translator.translate(texto, src='es', dest='en').text

# Función para traducir la respuesta al español
def traducir_a_espanol(texto):
    return translator.translate(texto, src='en', dest='es').text

# Función para generar la nube de palabras
def generar_nube_palabras(texto):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Función para realizar un análisis de clustering (Map-Reduce) con KMeans
def clustering_documento(texto):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([texto])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    return kmeans, X_pca

# Interfaz en Streamlit
st.title("RAG con Streamlit y GitHub")

# Cargar PDF desde la URL o cargar un PDF proporcionado por el usuario
uploaded_file = st.file_uploader("Cargar un archivo PDF", type=["pdf"])

if uploaded_file is not None:
    with open("user_uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Archivo PDF cargado con éxito.")
    texto = extraer_texto("user_uploaded_pdf.pdf")
    indexar_texto(texto)
else:
    st.write("Procesando PDF predeterminado...")
    pdf_path = descargar_pdf(GITHUB_PDF_URL)
    if pdf_path:
        texto = extraer_texto(pdf_path)
        indexar_texto(texto)
        st.success("PDF procesado e indexado correctamente.")

# Mostrar vista previa del texto extraído
st.subheader("Vista previa del PDF procesado (primeros 1000 caracteres):")
st.text(texto[:1000] if texto else "No se ha podido extraer texto del documento.")

# Depuración: Mensaje para asegurar que el código llega a este punto
st.write("Renderizando botón de estadísticas...")  

# Botón SIEMPRE visible para ver estadísticas
if st.button("Ver estadísticas del documento"):
    if texto:
        # Mostrar estadísticas
        st.subheader("Estadísticas del Documento")
        num_palabras = len(texto.split())
        st.write(f"Número de palabras: {num_palabras}")

        num_oraciones = len(texto.split(". "))
        st.write(f"Número de oraciones: {num_oraciones}")

        num_parrafos = len(texto.split("\n"))
        st.write(f"Número de párrafos: {num_parrafos}")

        # Generar nube de palabras
        st.subheader("Nube de Palabras")
        generar_nube_palabras(texto)

        # Análisis de clustering (Map-Reduce)
        st.subheader("Clustering del Documento (Map-Reduce)")
        kmeans, X_pca = clustering_documento(texto)
        st.write("Clustering (KMeans) y reducción de dimensionalidad con PCA realizada. Puedes ver la visualización a continuación.")

        # Mostrar gráfico de PCA
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
        ax.set_title("Visualización de Clustering con PCA")
        st.pyplot(fig)
    else:
        st.write("No se ha podido extraer texto del documento.")

