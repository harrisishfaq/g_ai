from langchain_ollama import OllamaLLM
def main():
    llm = OllamaLLM(model="gemma3:270m", temperature=0)
    user_query = input("Enter your query: ")

    response = llm.invoke(user_query)
    print(response)

if __name__ == "__main__":
    main()