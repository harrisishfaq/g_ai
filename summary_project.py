from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage

def main():
    print("Starting summary_project.py")

    question = input("Do you want random topic summary or of your own choice. For choice please press 1: ")

    if question.strip() == "1":
        user_input = input("Please enter the topic you want a summary of: ")
    else:
        user_input = "Artificial Intelligence and its impact on modern society."


    llm = ChatOllama(model="gemma3:270m", temperature=0)

    ##### METHOD 1 ####
    system_message = SystemMessage(content="You are a helpful assistant that summarizes topics.")
    user_message = HumanMessage(content=f"Please provide a concise summary of 50 words of the following topic: {user_input}")
    ai_message  = llm.invoke([system_message, user_message])

    print(ai_message.content)

    ##### METHOD 2 #####
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant that summarizes topics."},
    #     {"role": "user", "content": f"Please provide a concise summary of 50 words of the following topic: {user_input}"}
    # ]

    # response = llm.invoke(messages)
    # print(response.content)

if __name__ == "__main__":
    main()