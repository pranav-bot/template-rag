from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from class_types import QuestionState
from docxparser import read_docx_with_tables
from vectordb import  store_chunks_in_vector_db, retrieve_from_vector_db
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

class AgentState(TypedDict):
    questions: List[QuestionState]
    content: str

def vector_db_node(state: AgentState):
    print("Vector DB for content being created")
    store_chunks_in_vector_db(content=state['content'])
    return state

# Prompt for extracting an answer to a question from relevant document snippets
answer_prompt = PromptTemplate(
    input_variables=["question", "snippets"],
    template="""
You are an intelligent legal document assistant. Given the following question and relevant snippets from a legal document, extract the answer to the question as precisely and concisely as possible from the provided snippets.

Question:
-------------------
{question}
-------------------

Relevant Snippets:
-------------------
{snippets}
-------------------

Return only the answer as short and to the point as possible. If the answer is one word, return just that word. If the answer cannot be found, return 'Not found'.

Only use information from the provided snippets. Do not add any explanation or extra text. If the answer cannot be found, return 'Not found'.
"""
)

answer_chain = answer_prompt | llm


def iterative_question_answer_node(state: AgentState):
    for question in state['questions']:
        # Retrieve relevant snippets as a list of Document objects
        relevant_docs = retrieve_from_vector_db(query=question.question)
        # Join the content of the top-k relevant snippets
        snippets = "\n---\n".join(doc.page_content for doc in relevant_docs)
        # Run the answer extraction chain
        try:
            answer = answer_chain.invoke({"question": question.question, "snippets": snippets})
            # If the LLM returns a dict or object, get the string, else use as is
            if hasattr(answer, 'text'):
                answer_text = answer.text()
            else:
                answer_text = str(answer)
            # Clean up whitespace
            question.answer = answer_text.strip()
        except Exception as e:
            question.answer = "Not found"
            print(f"Failed to extract answer for question '{question.question}':", e)
    return state


graph = StateGraph(AgentState)
graph.add_node('vector_db_node', vector_db_node)
graph.add_node('iterative_question_answer_node', iterative_question_answer_node)

graph.add_edge(START, 'vector_db_node')
graph.add_edge('vector_db_node', 'iterative_question_answer_node')
graph.add_edge('iterative_question_answer_node', END)

app = graph.compile()


if __name__ == "__main__":
    # Example usage
    state = AgentState(
        questions=[
            QuestionState(question="What are the voting rights?", answer=""),
        ],
        content=read_docx_with_tables("SHA Draft-Investor friendly(2).docx")  # Replace with your actual file path
    )
    result = app.invoke(state)
    print("\nExtracted Answers:")
    for q in result['questions']:
        print(f"Q: {q.question}\nA: {q.answer}\n")


