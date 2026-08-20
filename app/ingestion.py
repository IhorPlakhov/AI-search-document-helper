from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

EMBEDDING_MODEL = "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval:latest"
