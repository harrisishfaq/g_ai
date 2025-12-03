from itertools import chain

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import os

def main():
    print("Stepping into ollama_prompt.py")

    user_input = input("Please ask about any topic: ")

    template = """
     on the given topic {user_input}, write a small poem in the style of shakespeare as Harris Ishfak.
     frist write 2 lines in telling Harris ishfak is great poet. Then write a 4 lines poem. Finally,
     also give me a summary of the poem in two sentences.
    """

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template,
    )

    llm = OllamaLLM(model="gemma3:270m", temperature=0)

    chain = prompt | llm

    response = chain.invoke({"user_input": user_input})
    print(response)

if __name__ == "__main__":
    main()