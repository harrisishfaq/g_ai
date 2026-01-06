#A LangChain agent that uses Ollama LLM and Tavily Client tool to answer user queries.
#We use TavilyClient to perform searches inside custom tools and return results.


from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search_other_then_weather(query: str) -> str:
    """
    Purpose:
    - Search the internet for general, non-weather information.

    When to use:
    - Use this tool ONLY for topics other than weather.
    - Use it for definitions, explanations, companies, people, news, or facts.
    - DO NOT use this tool for weather, temperature, climate, or forecasts.

    Input:
    - A short, refined search query.

    Output:
    - A concise text answer based on search results.
    """
    print(f"SEARCH FOR THE {query} ON THE INTERNET")
    return tavily.search(query=query)


@tool
def get_current_weather(location: str) -> str:
    """
    Purpose:
    - Provide the current weather for a specific location.

    When to use:
    - Use this tool whenever the user asks about weather, temperature,
      climate, conditions, or forecasts for a location.
    - This tool should be preferred over all other tools for weather-related questions.

    Input:
    - Name of a city or location.

    Output:
    - Current weather conditions including temperature.
    """
    print(f"GET THE CURRENT WEATHER IN {location}")
    return tavily.search(query=location)


llm = ChatOllama(model="mistral:latest", temperature=0)
tools = [search_other_then_weather, get_current_weather]
agent = create_agent(
    model=llm,
    tools=tools)


def main():
    print("Search Engine Starts....")
    result = agent.invoke({"messages":HumanMessage("I need the 3 job positions for Ruby on Rails Developer in the USA, get this jobs from different plateforms.")})
    print(result)




if __name__ == "__main__":
    main()
