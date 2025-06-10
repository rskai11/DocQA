from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from datetime import datetime
import re
import time
import logging
from session_management import RedisDB, MongoDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",  # React app on port 3000
    "http://localhost:5173",  # Vite app on port 5173
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base URL for the Ollama server
OLLAMA_SERVER_URL = "http://localhost:11434"

# Database connection classes with auto-reconnection
class ResilientRedisDB:
    def __init__(self, host="localhost", port=6379, password="rounak_test", max_retries=3):
        self.host = host
        self.port = port
        self.password = password
        self.max_retries = max_retries
        self.db = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.db = RedisDB(host=self.host, port=self.port, password=self.password)
            self.connected = self.db.connect()
            if self.connected:
                logger.info("Successfully connected to Redis")
            return self.connected
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.connected = False
            return False
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                if not self.connected:
                    logger.info(f"Redis not connected, attempting reconnection (attempt {attempt + 1})")
                    if not self._connect():
                        raise Exception("Failed to reconnect to Redis")
                
                return operation(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"Redis operation failed (attempt {attempt + 1}): {str(e)}")
                self.connected = False
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts")
                    raise e
    
    def insert(self, user_id, session_id, question, answer):
        """Insert data with auto-reconnection"""
        return self._retry_operation(self.db.insert, user_id=user_id, session_id=session_id, 
                                   question=question, answer=answer)
    
    def fetch(self, user_id=None, session_id=None):
        """Fetch data with auto-reconnection"""
        return self._retry_operation(self.db.fetch, user_id=user_id, session_id=session_id)

class ResilientMongoDB:
    def __init__(self, connection_string="mongodb://localhost:27017/", 
                 database_name="chat_history", collection_name="chat_messages",
                 username="rounak_admin", password="rounak_test", max_retries=3):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.db = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Initialize MongoDB connection"""
        try:
            self.db = MongoDB(
                connection_string=self.connection_string,
                database_name=self.database_name,
                collection_name=self.collection_name,
                username=self.username,
                password=self.password
            )
            self.connected = self.db.connect()
            if self.connected:
                logger.info("Successfully connected to MongoDB")
            return self.connected
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.connected = False
            return False
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                if not self.connected:
                    logger.info(f"MongoDB not connected, attempting reconnection (attempt {attempt + 1})")
                    if not self._connect():
                        raise Exception("Failed to reconnect to MongoDB")
                
                return operation(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"MongoDB operation failed (attempt {attempt + 1}): {str(e)}")
                self.connected = False
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"MongoDB operation failed after {self.max_retries} attempts")
                    raise e
    
    def insert(self, user_id, session_id, question, answer):
        """Insert data with auto-reconnection"""
        return self._retry_operation(self.db.insert, user_id=user_id, session_id=session_id,
                                   question=question, answer=answer)
    
    def fetch(self, user_id=None, session_id=None):
        """Fetch data with auto-reconnection"""
        return self._retry_operation(self.db.fetch, user_id=user_id, session_id=session_id)

# Initialize resilient database connections
db_redis = ResilientRedisDB(host="localhost", port=6379, password="rounak_test")
db_mongo = ResilientMongoDB(
    connection_string="mongodb://localhost:27017/",
    database_name="chat_history",
    collection_name="chat_messages",
    username="rounak_admin",
    password="rounak_test"
)

# Pydantic models
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

def load_prompt_template():
    """Load the prompt template from prompt.txt file"""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="prompt.txt file not found. Please create a prompt.txt file with {question} and {history} placeholders."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading prompt.txt: {str(e)}")

def extract_answer_from_response(response_text):
    """Extract the answer part after </think> tag or return full response if no think tag"""
    think_end_pattern = r'</think>\s*'
    match = re.search(think_end_pattern, response_text, re.IGNORECASE)
    
    if match:
        answer = response_text[match.end():].strip()
        return answer if answer else response_text
    else:
        return response_text.strip()

def format_conversation_history(session_records):
    """Format conversation history from Redis records into a readable string"""
    if not session_records:
        return "No previous conversation history."
    
    # Sort records by timestamp to maintain chronological order
    sorted_records = sorted(session_records, key=lambda x: x.get('timestamp', ''))
    
    history_lines = []
    for record in sorted_records:
        question = record.get('question', '')
        answer = record.get('answer', '')
        timestamp = record.get('timestamp', '')
        
        if question and answer:
            history_lines.append(f"User: {question}")
            history_lines.append(f"Assistant: {answer}")
            history_lines.append("---")
    
    # Remove the last separator
    if history_lines and history_lines[-1] == "---":
        history_lines.pop()
    
    return "\n".join(history_lines) if history_lines else "No previous conversation history."

def get_session_history(session_id):
    """Retrieve conversation history for a session from Redis"""
    try:
        session_records = db_redis.fetch(session_id=session_id)
        formatted_history = format_conversation_history(session_records)
        logger.info(f"Retrieved {len(session_records)} records for session {session_id}")
        return formatted_history
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        return "No previous conversation history available."

def store_chat_data_resilient(user_id: str, session_id: str, question: str, answer: str):
    """Store chat data in both Redis and MongoDB with resilient error handling"""
    redis_success = False
    mongo_success = False
    
    # Try to store in Redis
    try:
        redis_record_id = db_redis.insert(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer
        )
        logger.info(f"Successfully stored in Redis with ID: {redis_record_id}")
        redis_success = True
    except Exception as e:
        logger.error(f"Failed to store in Redis after all retries: {str(e)}")
    
    # Try to store in MongoDB
    try:
        mongo_record_id = db_mongo.insert(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer
        )
        logger.info(f"Successfully stored in MongoDB with ID: {mongo_record_id}")
        mongo_success = True
    except Exception as e:
        logger.error(f"Failed to store in MongoDB after all retries: {str(e)}")
    
    # Log storage status
    if redis_success and mongo_success:
        logger.info("Chat data successfully stored in both databases")
    elif redis_success or mongo_success:
        logger.warning(f"Chat data partially stored - Redis: {redis_success}, MongoDB: {mongo_success}")
    else:
        logger.error("Failed to store chat data in both databases")
    
    return {"redis_success": redis_success, "mongo_success": mongo_success}

@app.post("/create_session", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new session with user_id and return user_id, session_id, and timestamp."""
    try:
        session_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        return SessionResponse(
            user_id=request.user_id,
            session_id=session_id,
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.post("/ask_question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question with conversation history context and store the conversation with resilient database handling."""
    try:
        # Load prompt template from file
        prompt_template = load_prompt_template()
        
        # Get conversation history for this session from Redis
        conversation_history = get_session_history(request.session_id)
        
        # Replace placeholders in prompt template
        prompt = prompt_template.replace("{question}", request.question)
        prompt = prompt.replace("{history}", conversation_history)
        
        logger.info(f"Generated prompt with history for session {request.session_id}")
        
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
            raw_answer = ollama_response["response"]
            processed_answer = extract_answer_from_response(raw_answer)
            
            # Store chat data with resilient error handling
            storage_result = store_chat_data_resilient(
                user_id=request.user_id,
                session_id=request.session_id,
                question=request.question,
                answer=processed_answer
            )
            
            # Add storage status to logs but don't affect response
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
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama server. Make sure Ollama is running and qwen3:0.6b model is available."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{user_id}")
async def get_chat_history(user_id: str, session_id: str = None, source: str = "redis"):
    """Retrieve chat history with resilient database handling."""
    try:
        if source.lower() == "redis":
            if session_id:
                records = db_redis.fetch(session_id=session_id)
            else:
                records = db_redis.fetch(user_id=user_id)
            return {"source": "Redis", "records": records, "count": len(records)}
            
        elif source.lower() == "mongodb":
            if session_id:
                records = db_mongo.fetch(session_id=session_id)
            else:
                records = db_mongo.fetch(user_id=user_id)
            return {"source": "MongoDB", "records": records, "count": len(records)}
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source '{source}'. Use 'redis' or 'mongodb'"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.get("/health")
def health_check():
    """Enhanced health check with database connection status"""
    try:
        # Check Ollama server
        ollama_response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
        ollama_status = ollama_response.status_code == 200
        
        # Check if qwen3 model is available
        qwen_available = False
        if ollama_status:
            models = ollama_response.json().get("models", [])
            qwen_available = any("qwen3:0.6b" in model.get("name", "") for model in models)
        
        # Check prompt file
        prompt_exists = os.path.exists("prompt.txt")
        
        return {
            "ollama_status": "running" if ollama_status else "not running",
            "qwen3_available": qwen_available,
            "prompt_file_exists": prompt_exists,
            "redis_connected": db_redis.connected,
            "mongodb_connected": db_mongo.connected,
            "message": "All services healthy" if all([ollama_status, qwen_available, prompt_exists, db_redis.connected, db_mongo.connected]) else "Some services need attention"
        }
        
    except Exception as e:
        return {
            "ollama_status": "error",
            "error": str(e),
            "redis_connected": db_redis.connected,
            "mongodb_connected": db_mongo.connected
        }
