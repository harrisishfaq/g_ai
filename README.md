######
INSTALLATION
######
1: Langchain
2: uv

######
STEPS
######
uv init                         #FOR CREATE PROJECT DIRECTORY
uv add langchain
uv add python-dotenv 
uv add black isort              #FOR PYTHON FORMATTING
uv add langchain-ollama         #FOR OLLAMA LLM
uv add langchain-google-genai   #FOR GOOGLE GEMINI API


######
To Activate Ollama Server
######
1: install ollama from https://ollama.com/
2: run ollama server by command:
   ollama serve
3: If required, download models by command:
   ollama pull <model-name>
   Example:
   ollama pull llama2
   ollama pull gemini-pro
   ollama pull gemma3:270m

######
TO RUN SCRIPTS
######
uv run main.py
