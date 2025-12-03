from langchain_ollama import ChatOllama

def main():
    print("Starting summary_project.py")

    question = input("Do you want random topic summary or of your own choice. For choice please press 1: ")

    if question.strip() == "1":
        user_input = input("Please enter the topic you want a summary of: ")
    else:
        user_input = "Artificial Intelligence and its impact on modern society."


    llm = ChatOllama(model="gemma3:270m", temperature=0)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes topics."},
        {"role": "user", "content": f"Please provide a concise summary of 50 words of the following topic: {user_input}"}
    ]

    response = llm.invoke(messages)
    print("Summary:")
    print(response.content)

if __name__ == "__main__":
    main()