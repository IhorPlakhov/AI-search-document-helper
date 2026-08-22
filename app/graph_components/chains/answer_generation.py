from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-3.7-flash"
llm = ChatGoogleGenerativeAI(model=MODEL)

system_prompt = """
You are an expert AI assistant who answers questions based on the provided documents.

Answer the question using only the context from the retrieved documents.
If the answer is not in the documents, reply with "Answer not found in the provided documents."
Do not use any external knowledge. Do not use any information outside the provided documents.
Do not format the answer using markdown.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Question: {question}\n Retrieved Documents:\n {documents}"),
    ]
)

response_chain = prompt | llm | StrOutputParser()
