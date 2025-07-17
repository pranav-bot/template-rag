from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docxparser import read_docx_with_tables

# content = read_docx_with_tables("SHA Draft-Investor friendly(2).docx")  # Replace with your actual file path


def split_document(content: str, chunk_size=2000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(content)

# print(split_document(content)[1])