# this file holds complete RAG pipeline code
# ingestion -> RAG querying

# for ingestion we need -> ["Document Loaders", "Text Splitters", "Embeddings", "Vector Store"]
# for querying we need -> ["LLM", "Prompting", "Retrieval Chain", "Document Stuffing", "QA Chain"]

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import hashlib


load_dotenv()


def main():
    print("1st step is ingestion")


    print("INGESTION 1st step: Document loaders...")
    pdf_loader = PyPDFLoader("/Users/harrisishfaq/Desktop/PycharmProjects/g_ai/cheatsheet.pdf")
    docs = pdf_loader.load()


    print("INGESTION 2nd step: Splitting...")
    splitting = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    pdf_chunks = splitting.split_documents(docs)
    print(f"Loaded {len(pdf_chunks)} chunks")


    print("INGESTION 3rd step: Embedding...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


    if False: # Change to True if you want to re-ingest documents
        vector_store = PineconeVectorStore.from_documents(pdf_chunks, embeddings, index_name="pdfstore")
    else:
        vector_store = PineconeVectorStore(embedding=embeddings, index_name="pdfstore")


    print("2nd step is RAG Querying")
    print("RAG Querying starts...")

    llm =OllamaLLM(model="mistral:latest", temperature=0)
    # llm = OllamaLLM(model="gemma3:270m", temperature=0)

    user_question = input("Ask me anything about the story: ")

    custom_prompt = True
    if custom_prompt:
        print("Using custom prompt template...")
        template = """
        Answer the question based ONLY on the context below.
        If you don't know the answer, say 'I don't know.'

        Context:
        {context}

        Question:
        {input}

        Answer:
        """

        prompt = PromptTemplate.from_template(template)
    else:
        print("Using ready-made prompt from hub...")
        prompt = hub.pull("langchain-ai/retrieval-qa-chat")




    combine_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), # It is retriever from vector store
        combine_docs_chain=combine_document_chain, # it is augmentation chain
    )

    result = retrieval_chain.invoke({"input": user_question}) # Now Its time to generate the answer from the retrieval chain
    print("Answer:", result['answer'].strip())


if __name__ == "__main__":
    print("RAG Process starts....")
    main()
    print("RAG Process ends....")
