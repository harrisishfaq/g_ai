# A LangChain agent that uses Ollama LLM and Tavily search tool to answer user queries.
# We don't use custom tools into it


from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaLLM
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch


llm = ChatOllama(model="mistral:latest", temperature=0)
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools)

def main():
    print("Hello from react_agent_tavily_search.py!")
    result = agent.invoke({"messages":HumanMessage("I need the 3 job positions for Ruby on Rails Developer in the USA, get this jobs from different plateforms.")})
    print(result);

if __name__ == "__main__":
    main()