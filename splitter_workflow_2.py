import os
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from class_types import Section
from docxparser import read_docx_with_tables
from pdfparser import read_pdf_with_tables
from utils import split_document
from vectordb import delete_vector_store, initialize_vector_db
from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 2000

class AgentState(TypedDict):
    document_path: str
    content: str
    content_chunks: list[str]
    temp_result: Section
    vector_store_name: str
    sections: dict[str, str]

def content_loader_node(state: AgentState) -> AgentState:
    print("Content loading node called")
    _, ext = os.path.splitext(state['document_path'])
    file_type = ext.lstrip('.').lower()
    if file_type == 'pdf':
        state['content'] = read_pdf_with_tables(state['document_path'])
    elif file_type == 'docx':
        state['content'] = read_docx_with_tables(state['document_path'])

    state['content_chunks'] = split_document(state['content'], chunk_size=CHUNK_SIZE)

    state['vector_store_name'] = _
    if os.path.exists(os.path.join("vector_indexes", _)):
        delete_vector_store(name=_)
        
    initialize_vector_db(name=_)
    return state


content_parse_prompt = PromptTemplate(
    input_variables=["content"],
    template="""You are a legal AI assistant. From the following contract text, extract the **specific section** (if any) with:
- Section title (e.g., "Confidentiality", "Termination", etc.)
- Full content of that section""")