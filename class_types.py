from pydantic import BaseModel, Field

class BelongsToSection(BaseModel):
    belongs: bool = Field(description="Indicates if the section belongs to the snippet or not")

class Section(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section")

class FieldExtractionResult(BaseModel):
    field: str = Field(description="Field name that needs to be filled")
    description: str = Field(description="Description of the field and its purpose")

class QuestionExtractionResult(BaseModel):
    question: str = Field(description="Question that needs to be answered")
    fields_it_fills: FieldExtractionResult
    description: str = Field(description="Description of the question and its purpose")

class QuestionState(BaseModel):
    question: str = Field(description="The question that needs to be answered")
    answer: str = Field(description="The answer extracted from the relevant snippets")

class Section2(BaseModel):
    title: str = Field(description="Title of the section")