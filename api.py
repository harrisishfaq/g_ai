from fastapi import FastAPI
from pydantic import BaseModel
import ollama_prompt_api
import asyncio


app = FastAPI(title="Simple Prompt API")

class PromptRequest(BaseModel):
    topic: str

class IndividualRequest(BaseModel):
    individual_id: int


@app.post("/generate")
def generate_poem(request: PromptRequest):
    result = ollama_prompt_api.main(request.topic)
    return {"result": result}

@app.post("/hello")
def hello_individual(request: IndividualRequest):
    return {"message": f"Hello, individual {request.individual_id}!"}


@app.post("/hello_org")
async def hello_individual(request: dict):
    return {"message": f"Hello, organization {request["org_id"]}!"}

@app.post("/test")
async def testing(request: dict):
    result = ollama_prompt_api.test()
    return {"message": result}
