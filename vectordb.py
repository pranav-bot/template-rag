from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Optional, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# 2. Retrieval function
def retrieve_from_vector_db(query: str, k: int = 5):
    # Load vectorstore from disk
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=k)
    return docs