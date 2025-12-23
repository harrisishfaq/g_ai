from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()

import os


def main(user_input):

    template = """
     on the given topic {user_input}, write a small poem in the style of shakespeare as Harris Ishfak.
     frist write 2 lines in telling Harris ishfak is great poet. Then write a 4 lines poem. Finally,
     also give me a summary of the poem in two sentences.
    """

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template,
    )

    # llm = OllamaLLM(model="gemma3:270m", temperature=0)
    llm = OllamaLLM(model="nomic - embed - text: latest", temperature=0)

    chain = prompt | llm

    response = chain.invoke({"user_input": user_input})
    return response


def test():
    return "I am loving it"

"""
Below is an example of how to call the main function with a sample input. it is only used when i run that script in terminal or by clicking run button.
it has no concern for api call.
"""
if __name__ == "__main__":
    main("nature")
