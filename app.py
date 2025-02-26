import os
import io
import json
import tempfile
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

app = Flask(__name__)

# --- Load API Keys from Environment Variables ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # Set this in your environment variables

# --- Load Credentials from Environment Variable ---
if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise ValueError("Missing Google Service Account JSON in environment variables.")

service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
credentials = service_account.Credentials.from_service_account_info(service_account_info)
drive_service = build("drive", "v3", credentials=credentials)

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
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=GOOGLE_API_KEY)
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
