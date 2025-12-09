import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def main():
    print("INGESTION:Document loaders...")
    text_loader = TextLoader("/Users/harrisishfaq/Desktop/PycharmProjects/g_ai/medium-blog.txt")
    documents = text_loader.load()

    print("INGESTION:Splitting...")
    spliting = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = spliting.split_documents(documents)
    print(f"Number of documents: {len(texts)}")

    print("INGESTION:Embedding...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    # embeddings = OllamaEmbeddings(model="all-minilm:latest")
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))


if __name__ == "__main__":
    print("Starting RAG Ingestion Process START")
    main()
    print("Starting RAG Ingestion Process END")