import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import load_credentials_from_file
import tempfile

app = Flask(__name__)

# --- Set API Keys ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyA4jwQI-BYjyc_qfqb_00mjm6nW2McIdlM"  # Replace with your actual API key

# --- Google Drive API Setup ---
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "credentials", "gemini-chatbot-project-450909-1d9e13bab354.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_ID = "1xqOpwgwUoiJYf9GkeuB4dayme4zJcujf"  # Replace with your folder ID

creds, _ = load_credentials_from_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=creds)

# --- Function to List Files in Google Drive ---
def get_files_from_drive():
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get("files", [])

# --- Function to Download a File from Google Drive ---
def download_file(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    file_stream.seek(0)
    return file_stream

# --- Function to Load Documents ---
def load_documents(file_stream, mime_type):
    if mime_type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_stream.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        return documents

    elif mime_type == "text/plain":
        loader = TextLoader(file_stream)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_stream)
    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or DOCX.")

    return loader.load()

# --- Function to Split Documents into Chunks ---
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# --- Function to Create Vector Store Using FAISS ---
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return FAISS.from_documents(chunks, embeddings)

# --- Function to Query Document and Get Answer ---
def query_document(query, vector_store):
    retriever = vector_store.as_retriever()
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=os.environ["GOOGLE_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)
    return qa_chain.invoke(query)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html", files=get_files_from_drive())

@app.route("/query", methods=["POST"])
def process_query():
    data = request.json
    query = data.get("query")

    files = get_files_from_drive()
    if not files:
        return jsonify({"response": "No files found in Google Drive."})

    chosen_file = files[0]
    file_id = chosen_file["id"]
    mime_type = chosen_file["mimeType"]
    file_stream = download_file(file_id)
    documents = load_documents(file_stream, mime_type)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

    response = query_document(query, vector_store)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
