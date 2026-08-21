from chromadb.api.types import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma, PersistentClient
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

EMBEDDING_MODEL = "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval:latest"
DATABASEDIRECTORY = "./.chromadb"

loader = UnstructuredLoader("langchain_full_docs.txt")
# .load() transform UnstructuredLoader into list[Document]
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400,
    chunk_overlap=50,
)

doc_splits = text_splitter.split_documents(documents)


def add_doc_in_database(collection_name: str) -> None:
    """Add the documents in the database."""

    client = PersistentClient(path=DATABASEDIRECTORY)
    existing_collections = [col.name for col in client.list_collections()]

    if collection_name in existing_collections:
        Chroma.from_documents(
            documents=doc_splits,
            collection_name=collection_name,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=DATABASEDIRECTORY,
        )


def get_data_with_database(collection_name: str) -> VectorStoreRetriever:
    """Get the documents from the database."""

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=DATABASEDIRECTORY,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})

def retrieve_documents_node(state: GraphState, collection_name: str) -> dict[str, list[Document]]:
        """Search the documents for the given query and returns the updated 'documents' key for GraphState."""

        query = state["query"]
        documents = get_data_with_database(collection_name).invoke(query)

        return {"documents": documents}

