from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from answer_extractor import app as answer_app, AgentState
from class_types import QuestionState
from docxparser import read_docx_with_tables
import tempfile
import shutil
import os

class QuestionRequest(BaseModel):
    questions: List[str]

class AnswerResponse(BaseModel):
    question: str
    answer: str

app = FastAPI()

@app.post("/extract-answers", response_model=List[AnswerResponse])
def extract_answers(questions: QuestionRequest, file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    # Read content from docx
    content = read_docx_with_tables(tmp_path)
    # Prepare state
    state = AgentState(
        questions=[QuestionState(question=q, answer="") for q in questions.questions],
        content=content
    )
    result = answer_app.invoke(state)
    # Clean up temp file
    try:
        file.file.close()
        tmp.close()
    except Exception:
        pass
    shutil.os.remove(tmp_path)
    # Prepare response
    return [AnswerResponse(question=q.question, answer=q.answer) for q in result['questions']]
