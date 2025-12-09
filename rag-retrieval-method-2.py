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


    template = """
        You are the helpful assistant. Use the following pieces of context to answer the question at the end. if you dont know the answer
        simply say 'I don't know',  dont make up the answers. create 2 3 paragraphs of the answer and dont just give a one line answer.
        at every question asked say Great Question, Excellent Question or Wonderful Question at the start of your answer.
        {context}

    question: {input}
    Helpful answer:
    """

    # prompt = PromptTemplate(
    #     input_variables=["context", "input"],
    #     template=template,
    # )

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
         {"context": vector_store.as_retriever() | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm)

    res = rag_chain.invoke("What is embeddings? write a paragraphg")

    print(res)


if __name__ == "__main__":
    print ("RETRIEVAL PROCESS STARTS")
    main()
    print ("RETRIEVAL PROCESS ENDS")




