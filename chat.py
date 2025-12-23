from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()

def main():
    print("Starting interactive chat. Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Ending chat. Goodbye!")
            break

        # Add user input to history
        chat_history.append(f"User: {user_input}")

        # Combine chat history for context (simple concatenation)
        # You can improve by summarizing or using a window if too long
        context = "\n".join(chat_history)

        template = """
        You are a helpful assistant chatting with the user.

        Conversation history:
        {context}

        Now answer the user's latest question or comment:
        """

        prompt = PromptTemplate(
            input_variables=["context"],
            template=template,
        )

        llm = OllamaLLM(model="mistral:latest", temperature=0)

        chain = prompt | llm

        response = chain.invoke({"context": context})

        print("Bot:", response)

        # Add bot response to history
        chat_history.append(f"Bot: {response}")

if __name__ == "__main__":
    main()
