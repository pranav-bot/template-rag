import os
from typing import List, TypedDict, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from docxparser import read_docx_with_tables
from pdfparser import read_pdf_with_tables
from class_types import BelongsToSection, Section
from utils import split_document
from vectordb import delete_doc_from_vector_store, delete_vector_store, initialize_vector_db, retrieve_from_vector_db, store_section_in_vector_store


TOKEN_LIMIT = 4096

CHUNK_SIZE = 2000

CHUNK_OVERLAP = 200

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)


class AgentState(TypedDict):
    document_path: str
    content: str
    content_chunks: List[str]
    temp_result: Section
    vector_store_name: str
    sections: Dict[str, str]


def content_loader_node(state: AgentState) -> AgentState:
    print("Content loading node called")
    _, ext = os.path.splitext(state['document_path'])
    file_type = ext.lstrip('.').lower()
    if file_type == 'pdf':
        state['content'] = read_pdf_with_tables(state['document_path'])
    elif file_type == 'docx':
        state['content'] = read_docx_with_tables(state['document_path'])

    state['content_chunks'] = split_document(state['content'], chunk_size=200)

    state['vector_store_name'] = _
    if os.path.exists(os.path.join("vector_indexes", _)):
        delete_vector_store(name=_)
        
    initialize_vector_db(name=_)
    return state

section_parse_prompt = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are a legal AI assistant. From the following contract text chunk, extract the **specific section** (if any) with:
- Section title (e.g., "Confidentiality", "Termination", etc.)
- Full content of that section

If no section is found, return output as JSON with title fields as empty string ad content with the content:
{{
  "title": "",
  "content": ""
}}

Return output as JSON:
{{
  "title": "...",
  "content": "..."
}}

Chunk:
{chunk}
""")

section_parse_prompt_structure_parser = PydanticOutputParser(pydantic_object=Section)

section_parser_chain = section_parse_prompt | LLM | section_parse_prompt_structure_parser


def section_parser_node(state: AgentState) -> AgentState:
    print("Section parsing node called")
    print(len(state['content_chunks']), "chunks to process")
    for chunk in state['content_chunks']:
        print(f"Processing chunk: {chunk[:100]}...")
        state['temp_result'] = section_parser_chain.invoke({"chunk": chunk})
        ccs_bool = cross_check_section(state['temp_result'], vectorstore_name=state['vector_store_name'], state=state, overlap=CHUNK_OVERLAP)
        if state['temp_result'].title!="" and state['temp_result'].content:
            if not ccs_bool:
                store_section_in_vector_store(state['temp_result'], name=state['vector_store_name'])
                state['sections'].update({state['temp_result'].title: state['temp_result'].content})
            print(f"Stored section '{state['temp_result'].title}'")
        else:
            print("No valid section found in chunk, skipping.")
        
    
    return state

cross_check_prompt = PromptTemplate(
    input_variables=["section_content", "section_title","snippet", "snippet_title"],
    template="""
You are a legal AI assistant. Given the following section content and a snippet from the vector store, determine if the section content belongs to (is part of, or is a restatement of) the snippet. Consider legal meaning, context, and wording.

Section Content:
-------------------
{section_title}
{section_content}
-------------------

Snippet from Vector Store:
-------------------
{snippet_title}
{snippet}
-------------------

Return output as JSON. Do NOT wrap your output in markdown code blocks like ```json. Only return raw JSON:
{{
  "belongs": true|false
}}
"""
)

cross_check_parser = PydanticOutputParser(pydantic_object=BelongsToSection)

cross_check_chain = cross_check_prompt | LLM | cross_check_parser

def cross_check_section(result: Section, vectorstore_name: str, state: AgentState, overlap: int = 200) -> bool:
    """Check if the section title is already present in the list of sections."""
    relevant_docs = retrieve_from_vector_db(query=result.content, name=vectorstore_name)
    print(relevant_docs)
    most_relevant_snippet = relevant_docs[0] if relevant_docs else None
    u_id = most_relevant_snippet.id if most_relevant_snippet else None
    if not u_id:
        return False
    print(f"id: {u_id}")
    if most_relevant_snippet:
        belongs_result = cross_check_chain.invoke({
            "section_content": result.content,
            "section_title": result.title,
            "snippet": most_relevant_snippet.page_content,
            "snippet_title": most_relevant_snippet.metadata.get("title", "")
        })
        final_result = belongs_result.belongs
        
        print(f"final result: {final_result}")
        if final_result:
            delete_doc_from_vector_store(doc_id=u_id, name=vectorstore_name)
            temp_result = result
            temp_result.content = most_relevant_snippet.page_content + temp_result.content[CHUNK_OVERLAP:]
            store_section_in_vector_store(temp_result, name=vectorstore_name)
            state['sections'].update({most_relevant_snippet.metadata.get("title", ""): temp_result.content})
            return True
    return False

graph = StateGraph(AgentState)
graph.add_node('content_loader_node', content_loader_node)
graph.add_node('section_parser_node', section_parser_node)

graph.add_edge(START, 'content_loader_node')
graph.add_edge('content_loader_node', 'section_parser_node')
graph.add_edge('section_parser_node', END)

app = graph.compile()

if __name__ == "__main__":
    # Example usage
    initial_state = AgentState(
        document_path="testting_docs/NON-DISCLOSURE AGREEMENT (NDA)(1).docx",
        content="",
        content_chunks=[],
        temp_result=Section(title="", content=""),
        vector_store_name="",
        sections={}
    )
    final_state = app.invoke(initial_state)
    print(final_state['sections'])  # Output the extracted sections