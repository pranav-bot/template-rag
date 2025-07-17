import faiss
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Optional, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from utils import split_document
load_dotenv()


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 1. Accept text content directly, split, and store in vector DB
def store_chunks_in_vector_db(content: str, metadata: Optional[Dict] = None):
    chunks = split_document(content)
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    # Create vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embedding_model)
    # Save to disk
    vectorstore.save_local("faiss_index")

# 1. Accept text content directly, split, and store in vector DB
def store_chunks_in_vector_db_named(content: str, name: str , metadata: Optional[Dict] = None):
    chunks = split_document(content)
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    # Create vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embedding_model)
    # Save to disk
    vectorstore.save_local(name)

# 2. Retrieval function
def retrieve_from_vector_db(query: str, k: int = 5, name: str = "faiss_index"):
    # Load vectorstore from disk
    vectorstore = FAISS.load_local(name, embedding_model, allow_dangerous_deserialization=True)
    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def initialize_vector_db(name: str = "faiss_index"):
    """Initialize an empty FAISS vector database"""
    if not os.path.exists(name):
        # Get embedding dimension from the model
        test_embedding = embedding_model.embed_query("test")
        dimension = len(test_embedding)
        
        # Create empty FAISS index (using IndexFlatIP for cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        
        # Create empty vectorstore with the index
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),  # Empty document store
            index_to_docstore_id={}  # Empty mapping
        )
        
        # Save to disk
        vectorstore.save_local(name)
        print(f"Initialized empty vector DB '{name}'")
    else:
        print(f"Vector DB '{name}' already exists. No initialization needed.")

def add_doc_to_vector_db(content: str, name: str, metadata: Optional[Dict] = None):
    """Add a document to the specified vector database"""
    # Load vectorstore from disk
    vectorstore = FAISS.load_local(name, embedding_model, allow_dangerous_deserialization=True)
    # Create a document from the content
    document = Document(page_content=content, metadata=metadata or {})
    # Add the document to the vectorstore
    vectorstore.add_documents([document])
    # Save the updated vectorstore back to disk
    vectorstore.save_local(name)