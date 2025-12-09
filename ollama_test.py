from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

def main():
    llm = OllamaLLM(model="gemma3:270m", temperature=0)
    user_query = input("Enter your query: ")

    response = llm.invoke(user_query)
    print(response)

if __name__ == "__main__":
    main()