import time
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from class_types import FieldExtractionResult, QuestionExtractionResult
from pdfparser import read_pdf_with_tables
from docxparser import read_docx_with_tables
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
import json
from dotenv import load_dotenv
from pydantic import Field, BaseModel, RootModel
from langchain_core.output_parsers import PydanticOutputParser

from utils import split_document
# from vectordb import store_chunks_in_vector_db, retrieve_from_vector_db

load_dotenv()

TOKEN_LIMIT = 200



# For a list of fields, use RootModel (Pydantic v2)
FieldExtractionList = RootModel[list[FieldExtractionResult]]
# For a list of questions, use RootModel (Pydantic v2)
QuestionExtractionList = RootModel[list[QuestionExtractionResult]]

# Gemini model initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

class AgentState(TypedDict):
    file_path: str
    file_type: str
    content: str
    content_size: int
    content_chunks: Optional[List[str]]
    loop_count: int
    fields_to_be_filled: List[FieldExtractionResult]
    questions_needed_to_be_answered: List[str]
    abstract_questions: List[str]

def content_extraction_node(state: AgentState) -> AgentState:
    print(1)
    _, ext = os.path.splitext(state["file_path"])
    state["file_type"] = ext.lstrip(".").lower()

    if state["file_type"] == "pdf":
        state["content"] = read_pdf_with_tables(state["file_path"])
    elif state["file_type"] == "docx":
        state["content"] = read_docx_with_tables(state["file_path"])
    else:
        state["content"] = ""

    # store_chunks_in_vector_db(state["content"], metadata={
    #     "file_path": state["file_path"],
    #     "file_type": state["file_type"]
    # })

    # state["content_size"] = len(state["content"])

    return state

def splitter_node(state: AgentState) -> AgentState:
    print(2)
    if not state.get("content"):
        return state

    state["content_chunks"] = split_document(state["content"])
    return state

def router1_node(state: AgentState) -> str:
    print(4)
    if len(state['content']) > TOKEN_LIMIT:
        return "splitter_node"
    return "fields_extraction"




field_prompt = PromptTemplate(
    input_variables=["document"],
    template="""
You are an intelligent legal document assistant. Your task is to analyze the content of the following legal document and identify **all fields** that are expected to be **filled in by the user** (such as dates, names, amounts, locations, etc.).

Document:
-------------------
{document}
-------------------

For each identified field, return a dictionary with:
- `"field"`: The exact name or label of the field (or a clear description if no explicit name is present)
- `"description"`: A brief explanation of what type of value is expected and why it is required in the context of the document

Return a **list of dictionaries** in the following format:

[
  {{
    "field": "Effective Date",
    "description": "The date on which the agreement becomes legally binding."
  }},
  {{
    "field": "Party A Name",
    "description": "The full legal name of the first party entering the agreement."
  }},
  ...
]

Only include fields that clearly need to be filled in by the user. Do not include fields that are already completed or are generic clauses.
"""
)



# Use parser for a list of fields
field_parser = PydanticOutputParser(pydantic_object=FieldExtractionList)

# Use RunnableSequence style for LangChain >=0.1.17
field_chain = field_prompt | llm | field_parser

def fields_extraction_node(state: AgentState) -> AgentState:
    if not state.get("content"):
        return state
    
    if not state.get("content_chunks"):
        try:
            result = field_chain.invoke({"document": state["content"]})
        # result is a RootModel[list[FieldExtractionResult]]
            state["fields_to_be_filled"] = result.root
        except Exception as e:
            state["fields_to_be_filled"] = []
            print("Failed to parse structured output:", e)

    if state.get("content_chunks"):
        count = 0
        # If we have content chunks, we can process them
        for chunk in state["content_chunks"]:
            try:
                if count == 5:
                    time.sleep(70)
                    count = 0  # Simulate a delay for processing
                count += 1
                result = field_chain.invoke({"document": chunk})
                state["fields_to_be_filled"].extend(result.root)
            except Exception as e:
                print("Failed to parse structured output:", e)
    return state

def questions_extraction_node(state: AgentState) -> AgentState:
    print(5)
    return state

# Build the graph
graph = StateGraph(AgentState)

graph.add_node("content_extraction", content_extraction_node)
graph.add_node("fields_extraction", fields_extraction_node)
graph.add_node("splitter", splitter_node)
graph.add_node("router1_node", lambda state: state)

graph.add_edge(START, "content_extraction")
graph.add_edge("content_extraction", "router1_node")
graph.add_conditional_edges("router1_node", router1_node, {
    "splitter_node": "splitter",
    "fields_extraction": "fields_extraction"
})

graph.add_edge("splitter", "fields_extraction")

graph.add_edge("fields_extraction",END)

app = graph.compile()

# Run
if __name__ == "__main__":
    answer = app.invoke({
        "file_path": "SHA Draft-Investor friendly(2).docx",
        "file_type": "",
        "content": "",
        "content_size": 0,
        "content_chunks": None,
        "loop_count": 0,
        "fields_to_be_filled": [],
        "questions_needed_to_be_answered": [],
        "abstract_questions": []
    })
    print("\n🔍 Extracted Fields to be Filled:")
    fields = answer.get("fields_to_be_filled", [])
    for field in fields:
        print(f"- {field.field}: {field.description}")

    # Save to JSON
    fields_json = [field.model_dump() for field in fields]
    with open("extracted_fields.json", "w", encoding="utf-8") as f:
        json.dump(fields_json, f, ensure_ascii=False, indent=2)
    print("\nFields saved to extracted_fields.json")