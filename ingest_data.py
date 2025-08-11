import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import shutil

# --- Configuration ---
JSON_FILE_PATH = "publications.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db"

def load_and_split_publications(file_path: str) -> list[Document]:
    """Loads and splits the documents from the JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            publications_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file: {e}")
        return []

    langchain_documents = []
    for pub in publications_data:
        content = pub.get("publication_description", "")
        metadata = {
            'title': pub.get('title', 'No Title'),
            'id': pub.get('id', 'No ID')
        }
        document = Document(page_content=str(content), metadata=metadata)
        langchain_documents.append(document)

    # --- THE CRITICAL CHANGE ---
    # The chunk size is now smaller to ensure it fits within the LLM's context window.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Reduced from 1000
        chunk_overlap=50,      # Adjusted overlap
        length_function=len,
    )
    chunks = text_splitter.split_documents(langchain_documents)
    print(f"Successfully split documents into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings(chunks: list[Document]):
    """Creates embeddings and stores them in Chroma."""
    if not chunks:
        print("No chunks to process. Exiting.")
        return

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Remove old database directory if it exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Removing old database at {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Creating and persisting new vector store...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Vector store created successfully.")

if __name__ == "__main__":
    document_chunks = load_and_split_publications(JSON_FILE_PATH)
    create_and_store_embeddings(document_chunks)
    print("\nData ingestion complete.")