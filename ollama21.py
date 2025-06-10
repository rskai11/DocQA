from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Qwen3 Q&A API", description="Ask questions and get answers from Qwen3 model")

origins = [
    "http://localhost:3000",  # If your React app runs on port 3000
    "http://localhost:5173",  # If your React app (e.g. Vite) runs on port 5173
    # Add any other origins your frontend might use
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Base URL for the Ollama server
OLLAMA_SERVER_URL = "http://localhost:11434"

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str

def load_prompt_template():
    """Load the prompt template from prompt.txt file"""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="prompt.txt file not found. Please create a prompt.txt file with {QUESTION} placeholder."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading prompt.txt: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to Qwen3 Q&A API", "model": "qwen3:0.6b"}

@app.post("/ask", response_model=Answer)
async def ask_question(query: Question):
    try:
        # Load prompt template from file
        prompt_template = load_prompt_template()
        
        # Replace {QUESTION} placeholder with actual question
        prompt = prompt_template.replace("{question}", query.question)
        
        
        # Prepare the request to the Ollama API
        ollama_api_url = f"{OLLAMA_SERVER_URL}/api/generate"
        payload = {
            "model": "qwen3:0.6b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_k": 20,
                "top_p": 0.95,
                "presence_penalty": 1.5
            }
        }
        
        # Send the request to the Ollama API
        response = requests.post(ollama_api_url, json=payload)
        response.raise_for_status()
        
        if response.status_code == 200:
            ollama_response = response.json()
            return Answer(
                question=query.question,
                answer=ollama_response["response"]
            )
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail="Error in Ollama API response"
            )
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Cannot connect to Ollama server. Make sure Ollama is running and qwen3:0.6b model is available."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    try:
        # Check if Ollama server is running
        response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            qwen_available = any("qwen3:0.6b" in model.get("name", "") for model in models)
            
            # Also check if prompt.txt exists
            prompt_exists = os.path.exists("prompt.txt")
            
            return {
                "ollama_status": "running",
                "qwen3_available": qwen_available,
                "prompt_file_exists": prompt_exists,
                "message": "Service is healthy" if (qwen_available and prompt_exists) else "Check model availability and prompt.txt file"
            }
    except:
        return {"ollama_status": "not running", "message": "Ollama server is not accessible"}
