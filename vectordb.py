import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import Optional, Dict
import os
from langchain_community.docstore.in_memory import InMemoryDocstore
from class_types import Section
from utils import split_document
import shutil
from uuid import uuid4
load_dotenv()


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Directory for all vector store folders
VECTOR_INDEXES_DIR = "vector_indexes"

# 1. Accept text content directly, split, and store in vector DB
def store_chunks_in_vector_db(content: str, metadata: Optional[Dict] = None):
    os.makedirs(VECTOR_INDEXES_DIR, exist_ok=True)
    chunks = split_document(content)
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(os.path.join(VECTOR_INDEXES_DIR, "faiss_index"))

# 1. Accept text content directly, split, and store in vector DB with custom name
def store_chunks_in_vector_db_named(content: str, name: str , metadata: Optional[Dict] = None):
    os.makedirs(VECTOR_INDEXES_DIR, exist_ok=True)
    chunks = split_document(content)
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(os.path.join(VECTOR_INDEXES_DIR, name))

# 2. Retrieval function
def retrieve_from_vector_db(query: str, k: int = 5, name: str = "faiss_index"):
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def initialize_vector_db(name: str = "faiss_index"):
    os.makedirs(VECTOR_INDEXES_DIR, exist_ok=True)
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    if not os.path.exists(vectorstore_path):
        test_embedding = embedding_model.embed_query("test")
        dimension = len(test_embedding)
        index = faiss.IndexFlatIP(dimension)
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vectorstore.save_local(vectorstore_path)
        print(f"Initialized empty vector DB '{vectorstore_path}'")
    else:
        print(f"Vector DB '{vectorstore_path}' already exists. No initialization needed.")

def add_doc_to_vector_db(content: str, name: str, metadata: Optional[Dict] = None):
    os.makedirs(VECTOR_INDEXES_DIR, exist_ok=True)
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    document = Document(page_content=content, metadata=metadata or {})
    vectorstore.add_documents([document], id=[str(uuid4())])  # Use a unique ID for each document
    vectorstore.save_local(vectorstore_path)

def store_section_in_vector_store(section: Section, name: str):
    os.makedirs(VECTOR_INDEXES_DIR, exist_ok=True)
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    document = Document(page_content=section.content, metadata={"title": section.title})
    vectorstore.add_documents([document], id=[str(uuid4())])  # Use a unique ID for each document
    vectorstore.save_local(vectorstore_path)

def delete_vector_store(name: str):
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
        print(f"Deleted existing vector store: {vectorstore_path}")
    else:
        print(f"Vector store '{vectorstore_path}' does not exist.")

def delete_doc_from_vector_store(doc_id: str, name: str):
    vectorstore_path = os.path.join(VECTOR_INDEXES_DIR, name)
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.delete([doc_id])
        vectorstore.save_local(vectorstore_path)
        print(f"Deleted document with ID '{doc_id}' from vector store '{name}'")
    else:
        print(f"Vector store '{vectorstore_path}' does not exist.")