"""
Improved FastAPI application for Ollama with 1000 concurrent user support.
This version uses async I/O operations and proper connection pooling.
"""

import os
import re
import time
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Async libraries for I/O operations
import httpx
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.concurrency import asynccontextmanager

# Configure logging with async-friendly approach
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Pydantic Models -----

class SessionRequest(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    user_id: str
    session_id: str
    timestamp: str

class QuestionRequest(BaseModel):
    user_id: str
    session_id: str
    timestamp: str
    question: str

class AnswerResponse(BaseModel):
    user_id: str
    session_id: str
    timestamp: str
    question: str
    answer: str

# ----- Configuration -----

# Environment variables with defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "rounak_test")
REDIS_POOL_SIZE = int(os.getenv("REDIS_POOL_SIZE", "100"))
REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", "5"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "chat_history")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "chat_messages")
MONGO_USER = os.getenv("MONGO_USER", "rounak_admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "rounak_test")
MONGO_POOL_SIZE = int(os.getenv("MONGO_POOL_SIZE", "100"))

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

MAX_REQUEST_CONCURRENCY = int(os.getenv("MAX_REQUEST_CONCURRENCY", "1000"))

# ----- Database Connection Management -----

# Global variable to store the prompt template (loaded once at startup)
PROMPT_TEMPLATE = ""

# Context manager for application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager to handle startup and shutdown tasks.
    This loads resources at startup and properly closes connections at shutdown.
    """
    # Load prompt template at startup (once)
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            global PROMPT_TEMPLATE
            PROMPT_TEMPLATE = file.read()
            logger.info("Prompt template loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load prompt template: {str(e)}")
        raise RuntimeError(f"Could not load prompt template: {str(e)}")
    
    # Create Redis connection pool
    redis_pool = aioredis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        max_connections=REDIS_POOL_SIZE,
        decode_responses=True,
        socket_timeout=REDIS_TIMEOUT,
        socket_connect_timeout=REDIS_TIMEOUT,
        retry_on_timeout=True
    )
    app.state.redis_pool = redis_pool
    
    # Create MongoDB connection
    mongo_client = AsyncIOMotorClient(
        MONGO_URI,
        username=MONGO_USER,
        password=MONGO_PASSWORD,
        maxPoolSize=MONGO_POOL_SIZE,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000
    )
    app.state.mongo_client = mongo_client
    app.state.mongo_db = mongo_client[MONGO_DB]
    app.state.mongo_collection = app.state.mongo_db[MONGO_COLLECTION]
    
    # Create HTTP client with connection pooling
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(OLLAMA_TIMEOUT),
        limits=httpx.Limits(max_connections=100)
    )
    
    logger.info("Application startup complete")
    
    # Yield control to FastAPI
    yield
    
    # Cleanup on shutdown
    await app.state.http_client.aclose()
    await app.state.redis_pool.disconnect()
    app.state.mongo_client.close()
    logger.info("Application shutdown complete")

# ----- FastAPI App Initialization -----

app = FastAPI(lifespan=lifespan)

# Configure CORS
origins = [
    "http://localhost:3000",  # React app
    "http://localhost:5173",  # Vite app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Database Dependencies -----

async def get_redis():
    """Get Redis connection from pool."""
    redis = aioredis.Redis(connection_pool=app.state.redis_pool)
    try:
        yield redis
    finally:
        # No need to close as the connection returns to the pool automatically
        pass

async def get_mongo_collection():
    """Get MongoDB collection."""
    yield app.state.mongo_collection

async def get_http_client():
    """Get HTTP client for external API calls."""
    yield app.state.http_client

# ----- Helper Functions -----

def extract_answer_from_response(response_text: str) -> str:
    """Extract the answer part or return the full response."""
    think_end_pattern = r'\s*'
    match = re.search(think_end_pattern, response_text, re.IGNORECASE)
    if match:
        answer = response_text[match.end():].strip()
        return answer if answer else response_text
    else:
        return response_text.strip()

async def format_conversation_history(session_records: List[Dict[str, Any]]) -> str:
    """Format conversation history from records into a readable string."""
    if not session_records:
        return "No previous conversation history."
    
    # Sort records by timestamp
    sorted_records = sorted(session_records, key=lambda x: x.get('timestamp', ''))
    
    history_lines = []
    for record in sorted_records:
        question = record.get('question', '')
        answer = record.get('answer', '')
        
        if question and answer:
            history_lines.append(f"User: {question}")
            history_lines.append(f"Assistant: {answer}")
            history_lines.append("---")
    
    # Remove the last separator
    if history_lines and history_lines[-1] == "---":
        history_lines.pop()
    
    return "\n".join(history_lines) if history_lines else "No previous conversation history."

async def store_chat_data(
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
    redis: aioredis.Redis,
    mongo_collection
):
    """Store chat data in both Redis and MongoDB with error handling."""
    timestamp = datetime.now().isoformat()
    document = {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "question": question,
        "answer": answer
    }
    
    redis_success = False
    mongo_success = False
    
    # Store in Redis
    try:
        # Use pipeline for atomic operations
        async with redis.pipeline() as pipe:
            # Add to user-specific list
            await pipe.lpush(f"user:{user_id}", document)
            # Add to session-specific list
            await pipe.lpush(f"session:{session_id}", document)
            # Execute pipeline
            await pipe.execute()
        redis_success = True
    except Exception as e:
        logger.error(f"Failed to store in Redis: {str(e)}")
    
    # Store in MongoDB
    try:
        await mongo_collection.insert_one(document)
        mongo_success = True
    except Exception as e:
        logger.error(f"Failed to store in MongoDB: {str(e)}")
    
    return {
        "redis_success": redis_success,
        "mongo_success": mongo_success,
        "timestamp": timestamp
    }

# ----- API Endpoints -----

@app.post("/create_session", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new session with user_id and return session details."""
    try:
        session_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        return SessionResponse(
            user_id=request.user_id,
            session_id=session_id,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.post("/ask_question", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    redis: aioredis.Redis = Depends(get_redis),
    mongo_collection = Depends(get_mongo_collection),
    http_client: httpx.AsyncClient = Depends(get_http_client)
):
    """
    Process a question with async operations for improved concurrency.
    This endpoint can handle many concurrent requests efficiently.
    """
    try:
        # Get conversation history from Redis
        session_records = await redis.lrange(f"session:{request.session_id}", 0, -1)
        # Convert from JSON strings to Python objects
        session_records = [eval(record) for record in session_records] if session_records else []
        
        # Format conversation history
        conversation_history = await format_conversation_history(session_records)
        
        # Use global prompt template (loaded at startup)
        prompt = PROMPT_TEMPLATE.replace("{question}", request.question)
        prompt = prompt.replace("{history}", conversation_history)
        
        # Prepare the request to the Ollama API
        ollama_api_url = f"{OLLAMA_SERVER_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_k": 20,
                "top_p": 0.95,
                "presence_penalty": 1.5
            }
        }
        
        # Send the request to the Ollama API asynchronously
        response = await http_client.post(ollama_api_url, json=payload)
        response.raise_for_status()
        
        if response.status_code == 200:
            ollama_response = response.json()
            raw_answer = ollama_response["response"]
            processed_answer = extract_answer_from_response(raw_answer)
            
            # Store chat data asynchronously
            storage_result = await store_chat_data(
                user_id=request.user_id,
                session_id=request.session_id,
                question=request.question,
                answer=processed_answer,
                redis=redis,
                mongo_collection=mongo_collection
            )
            
            logger.info(f"Storage status: {storage_result}")
            
            return AnswerResponse(
                user_id=request.user_id,
                session_id=request.session_id,
                timestamp=request.timestamp,
                question=request.question,
                answer=processed_answer
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Error in Ollama API response"
            )
            
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama server. Make sure Ollama is running and {OLLAMA_MODEL} model is available."
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{user_id}")
async def get_chat_history(
    user_id: str, 
    session_id: Optional[str] = None,
    redis: aioredis.Redis = Depends(get_redis),
    mongo_collection = Depends(get_mongo_collection)
):
    """Retrieve chat history with resilient database handling."""
    try:
        # Try Redis first
        if session_id:
            redis_records = await redis.lrange(f"session:{session_id}", 0, -1)
            redis_records = [eval(record) for record in redis_records] if redis_records else []
        else:
            redis_records = await redis.lrange(f"user:{user_id}", 0, -1)
            redis_records = [eval(record) for record in redis_records] if redis_records else []
        
        # If Redis fails or empty, fall back to MongoDB
        if not redis_records:
            logger.info("No records found in Redis, falling back to MongoDB")
            if session_id:
                mongo_records = await mongo_collection.find({"session_id": session_id}).to_list(length=100)
            else:
                mongo_records = await mongo_collection.find({"user_id": user_id}).to_list(length=100)
            return {"source": "MongoDB", "records": mongo_records, "count": len(mongo_records)}
        
        return {"source": "Redis", "records": redis_records, "count": len(redis_records)}
            
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check with database connection status."""
    try:
        # Create temporary connections to check health
        redis = None
        mongo = None
        http_client = None
        
        try:
            # Check Redis
            redis = aioredis.Redis(connection_pool=app.state.redis_pool)
            redis_status = await redis.ping()
        except Exception as e:
            redis_status = False
            logger.error(f"Redis health check failed: {str(e)}")
        
        try:
            # Check MongoDB
            mongo = app.state.mongo_client
            mongo_status = await mongo.admin.command('ping')
            mongo_status = mongo_status.get("ok") == 1
        except Exception as e:
            mongo_status = False
            logger.error(f"MongoDB health check failed: {str(e)}")
            
        try:
            # Check Ollama API
            http_client = app.state.http_client
            ollama_response = await http_client.get(f"{OLLAMA_SERVER_URL}/api/tags")
            ollama_status = ollama_response.status_code == 200
            
            # Check if model is available
            if ollama_status:
                models = ollama_response.json().get("models", [])
                model_available = any(OLLAMA_MODEL in model.get("name", "") for model in models)
            else:
                model_available = False
        except Exception as e:
            ollama_status = False
            model_available = False
            logger.error(f"Ollama health check failed: {str(e)}")
            
        # Check prompt file
        prompt_exists = bool(PROMPT_TEMPLATE)
        
        return {
            "redis_status": "running" if redis_status else "not running",
            "mongodb_status": "running" if mongo_status else "not running",
            "ollama_status": "running" if ollama_status else "not running",
            "model_available": model_available,
            "prompt_template_loaded": prompt_exists,
            "max_concurrency": MAX_REQUEST_CONCURRENCY,
            "message": "All services healthy" if all([redis_status, mongo_status, ollama_status, model_available, prompt_exists]) else "Some services need attention"
        }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ----- Rate Limiting Middleware -----

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """
    Middleware to implement basic rate limiting.
    For more sophisticated rate limiting, consider using a dedicated service.
    """
    # Implement a simple semaphore-based concurrency limiter
    from asyncio import Semaphore
    
    # Create semaphore if it doesn't exist
    if not hasattr(app.state, "request_semaphore"):
        app.state.request_semaphore = Semaphore(MAX_REQUEST_CONCURRENCY)
    
    # Try to acquire the semaphore
    if not app.state.request_semaphore.locked():
        async with app.state.request_semaphore:
            response = await call_next(request)
            return response
    else:
        # Return 429 Too Many Requests if the semaphore is full
        return HTTPException(
            status_code=429, 
            detail="Too many requests. Please try again later."
        )

# ----- Main Entry Point -----

if __name__ == "__main__":
    import uvicorn
    
    # Calculate optimal number of workers based on CPU cores
    import multiprocessing
    workers = min(multiprocessing.cpu_count() + 1, 8)  # Cap at 8 workers
    
    # Start uvicorn with optimized settings
    uvicorn.run(
        "improved-ollama:app", 
        host="0.0.0.0", 
        port=8000,
        workers=workers,
        log_level="info",
        loop="uvloop",  # Use uvloop for better performance
        http="httptools",  # Use httptools for better HTTP parsing
        limit_concurrency=1100,  # Slightly higher than our expected max
        limit_max_requests=10000,  # Restart worker after this many requests (for memory leaks)
        timeout_keep_alive=5,  # Reduce idle connection timeout
    )