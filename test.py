from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from class_types import Section
from langchain_core.output_parsers import PydanticOutputParser

from docxparser import read_docx_with_tables

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.3,
)

section_parse_prompt = PromptTemplate(
    input_variables=["document"],
    template="""
You are a legal AI assistant. Analyze the following legal document and split its content into sections based on headings and their corresponding content. Each section should include:
- A heading (e.g., "Confidentiality", "Termination", etc.)
- The full content under that heading.

If no headings are found, return output as JSON with the title field as an empty string and the content field containing the entire document.

Return the output as a list of JSON objects, where each object represents a section:
[
  {{
    "title": "Heading 1",
    "content": "Content under Heading 1"
  }},
  {{
    "title": "Heading 2",
    "content": "Content under Heading 2"
  }},
  ...
]

Document:
{document}
"""
)


section_parse_prompt_structure_parser = PydanticOutputParser(pydantic_object=Section)

section_parser_chain = section_parse_prompt | LLM 
path = "testting_docs/SHA Draft-Investor friendly(2).docx"  # Replace with your actual file path
full_text = read_docx_with_tables(path)

result = section_parser_chain.invoke({"document": full_text})

print(result)