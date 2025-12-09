import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def main():
    llm = OllamaLLM(model="gemma3:270m", temperature=0)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vector_store = PineconeVectorStore(embedding= embeddings, index_name=os.environ.get("INDEX_NAME"))

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_document_chain,
    )


    result = retrieval_chain.invoke({"input" : "What is pinecone vector database? write a paragraphg"})
    print(result['answer'].strip())


if __name__ == "__main__":
    print ("RETRIEVAL PROCESS STARTS")
    main()
    print ("RETRIEVAL PROCESS ENDS")




