import os
import time
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import tempfile

# --- Set API Keys ---
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your API key
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gemini-chatbot-project-450909-1d9e13bab354.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_ID = "1xqOpwgwUoiJYf9GkeuB4dayme4zJcujf"

# --- Google Drive API Setup ---
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=creds)

# --- Function to List Files in Google Drive Folder ---
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

# --- Function to Process Files ---
def process_documents():
    files = get_files_from_drive()
    if not files:
        print("❌ No files found in Google Drive folder.")
        return
    
    print(f"📂 Found {len(files)} files. Processing first file: {files[0]['name']}")
    
    file_id = files[0]["id"]
    mime_type = files[0]["mimeType"]
    file_stream = download_file(file_id)
    
    # Load document
    if mime_type == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(file_stream.read())
            temp_file.flush()
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
    elif mime_type == "text/plain":
        loader = TextLoader(file_stream)
        documents = loader.load()
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_stream)
        documents = loader.load()
    else:
        print("Unsupported file format.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# --- Main Function (Runs in Loop) ---
def main():
    vector_store = process_documents()
    if not vector_store:
        return

    retriever = vector_store.as_retriever()
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=os.environ["GOOGLE_API_KEY"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
    )

    while True:
        query = "Give me a summary of the document"  # Change this query based on your needs
        response = qa_chain.run(query)
        print("\n🤖 Bot Response:", response)

        # Sleep for 1 hour before running again (to keep the app running)
        time.sleep(3600)

if __name__ == "__main__":
    main()
