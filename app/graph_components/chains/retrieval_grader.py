from app.graph_components.graph_state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

MODEL = "gemini-3.7-flash"

llm = ChatGoogleGenerativeAI(model=MODEL)


class DocumentCompliance(BaseModel):
    """Checks if the retrieved documents are relevant to the user's query."""

    reason: str = Field(description="The reason for the decision.")
    is_relevant: bool = Field(
        description="True if the documents are relevant to the query, False otherwise."
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that checks if the retrieved documents are relevant to the user's query.",
        ),
        ("human", "Question: {question}\n\nRetrieved Documents:\n\n{documents}"),
    ]
)

retrieval_grader_chain = prompt | llm.with_structured_output(DocumentCompliance)


def retrieval_grader_node(state: GraphState) -> dict[str, str]:
    """Check if the retrieved documents are relevant to the user's query."""

    documents = "\n\n".join(state["documents"])

    compliance = retrieval_grader_chain.invoke(
        {"question": state["question"], "documents": documents}
    )

    return {"compliance": compliance}
