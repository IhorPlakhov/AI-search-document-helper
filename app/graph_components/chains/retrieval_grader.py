from app.graph_components.graph_state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

MODEL = "gemini-3.5-flash-lite"

llm = ChatGoogleGenerativeAI(model=MODEL)


class DocumentCompliance(BaseModel):
    """Checks if the retrieved documents are relevant to the user's query."""

    reason: str = Field(description="The reason for the decision.")
    is_relevant: bool = Field(
        description="True if the documents are relevant to the query, False otherwise."
    )


system_prompt = """
You are an expert retrieval grader evaluating whether retrieved documents are relevant to a user question.

Evaluation Rules:
1. Meaning over keywords: Look for semantic relevance and useful context that helps answer the question, not just matching keywords.
2. Partial match: If ANY part of the provided documents contains relevant facts, information, or partial answers to the question, grade it as relevant.
3. Irrelevant: If the documents are completely off-topic, contradictory noise, or lack any factual relation to the question, grade as not relevant.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Question: {question}\n\nRetrieved Documents:\n\n{documents}"),
    ]
)

retrieval_grader_chain = prompt | llm.with_structured_output(DocumentCompliance)


def is_retrieval_relevant(state: GraphState) -> bool:
    """Check if the retrieved documents are relevant to the user's query."""

    documents = "\n\n".join(state["documents"])

    compliance = retrieval_grader_chain.invoke(
        {"question": state["question"], "documents": documents}
    )

    return compliance.is_relevant
