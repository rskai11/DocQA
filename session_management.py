import redis
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
import pymongo
from pymongo import MongoClient
try:
    from pymongo.errors import AuthenticationError
except ImportError:
    try:
        from pymongo.errors import AuthenticationFailure as AuthenticationError
    except ImportError:
        # Fallback for very old versions
        from pymongo.errors import OperationFailure as AuthenticationFailure
from pymongo.errors import (
    ConnectionFailure, 
    ServerSelectionTimeoutError,
    OperationFailure
)
class RedisDB:
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 password: str = None, db: int = 0):
        """Initialize Redis DB configuration"""
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.redis_client = None
        self.connected = False
        self.table_name = "chat_messages"
    
    def connect(self) -> bool:
        """Connect to Redis database"""
        try:
            # Try connecting with password if provided
            if self.password:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    decode_responses=True,
                    socket_timeout=10,
                    socket_connect_timeout=5
                )
            else:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_timeout=10,
                    socket_connect_timeout=5
                )
            
            # Test the connection
            self.redis_client.ping()
            self.connected = True
            auth_status = "with authentication" if self.password else "without authentication"
            print(f"✅ Successfully connected to Redis {auth_status}")
            return True
            
        except redis.AuthenticationError as e:
            print(f"❌ Authentication failed: {e}")
            if self.password:
                print("🔄 Retrying without authentication...")
                try:
                    self.redis_client = redis.Redis(
                        host=self.host,
                        port=self.port,
                        db=self.db,
                        decode_responses=True,
                        socket_timeout=10,
                        socket_connect_timeout=5
                    )
                    self.redis_client.ping()
                    self.connected = True
                    print("✅ Connected to Redis without authentication")
                    return True
                except Exception as retry_error:
                    print(f"❌ Failed to connect without authentication: {retry_error}")
                    return False
            return False
            
        except redis.ConnectionError as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Make sure Redis is running")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def _check_connection(self) -> bool:
        """Check if connected to Redis"""
        if not self.connected or not self.redis_client:
            print("❌ Not connected to Redis. Please call connect() first.")
            return False
        return True
    
    def insert(self, user_id: str, session_id: str, question: str, answer: str) -> Optional[str]:
        """Insert data into Redis"""
        if not self._check_connection():
            return None
        
        try:
            # Generate unique record ID
            record_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            
            # Create record data
            record_data = {
                "id": record_id,
                "user_id": user_id,
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "timestamp": current_time,
                "created_at": current_time
            }
            
            # Define Redis keys
            record_key = f"table:{self.table_name}:record:{record_id}"
            user_index_key = f"table:{self.table_name}:index:user:{user_id}"
            session_index_key = f"table:{self.table_name}:index:session:{session_id}"
            user_session_index_key = f"table:{self.table_name}:index:user_session:{user_id}:{session_id}"
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Store the main record
            pipe.hmset(record_key, record_data)
            
            # Create indexes for efficient querying
            pipe.sadd(user_index_key, record_id)
            pipe.sadd(session_index_key, record_id)
            pipe.sadd(user_session_index_key, record_id)
            
            # Execute all operations
            results = pipe.execute()
            
            if all(results):
                print(f"✅ Inserted record with ID: {record_id}")
                return record_id
            return None
            
        except Exception as e:
            print(f"❌ Failed to insert record: {e}")
            return None
    
    def fetch(self, user_id: str = None, session_id: str = None, 
              limit: int = None) -> List[Dict]:
        """Fetch data from Redis"""
        if not self._check_connection():
            return []
        
        try:
            record_ids = set()
            
            # Determine which index to use based on parameters
            if user_id and session_id:
                # Fetch by user and session
                index_key = f"table:{self.table_name}:index:user_session:{user_id}:{session_id}"
                record_ids = self.redis_client.smembers(index_key)
            elif user_id:
                # Fetch by user only
                index_key = f"table:{self.table_name}:index:user:{user_id}"
                record_ids = self.redis_client.smembers(index_key)
            elif session_id:
                # Fetch by session only
                index_key = f"table:{self.table_name}:index:session:{session_id}"
                record_ids = self.redis_client.smembers(index_key)
            else:
                # Fetch all records (get all record keys)
                all_keys = self.redis_client.keys(f"table:{self.table_name}:record:*")
                record_ids = [key.split(':')[-1] for key in all_keys]
            
            if not record_ids:
                print("📭 No records found")
                return []
            
            # Fetch actual records
            records = []
            for record_id in record_ids:
                record_key = f"table:{self.table_name}:record:{record_id}"
                record_data = self.redis_client.hgetall(record_key)
                if record_data:
                    records.append(record_data)
            
            # Sort by timestamp (newest first)
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Apply limit if specified
            if limit and limit > 0:
                records = records[:limit]
            
            print(f"✅ Retrieved {len(records)} records")
            return records
            
        except Exception as e:
            print(f"❌ Failed to fetch records: {e}")
            return []
    
    def fetch_qa_pairs(self, user_id: str = None, session_id: str = None, 
                       limit: int = None) -> List[Dict]:
        """Fetch only question-answer pairs"""
        records = self.fetch(user_id=user_id, session_id=session_id, limit=limit)
        
        qa_pairs = []
        for record in records:
            question = record.get('question')
            answer = record.get('answer')
            
            if question and answer:
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })
        
        return qa_pairs
    
    def close_connection(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.connected = False
            print("✅ Redis connection closed")





class MongoDB:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "chat_history",
                 collection_name: str = "chat_messages",
                 username: str = None, password: str = None):
        """Initialize MongoDB Chat Table with flexible authentication"""
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.username = username
        self.password = password
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to MongoDB database"""
        try:
            # Configure connection with authentication if provided
            if self.username and self.password:
                # Parse connection string and add credentials
                if "@" not in self.connection_string:
                    # Add credentials to connection string
                    base_url = self.connection_string.replace("mongodb://", "")
                    self.connection_string = f"mongodb://{self.username}:{self.password}@{base_url}"

            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )

            # Test the connection
            self.client.admin.command('ping')

            # Get database and collection
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]

            # Create indexes for efficient querying
            self._create_indexes()

            self.connected = True
            auth_status = "with authentication" if self.username else "without authentication"
            print(f"✅ Successfully connected to MongoDB {auth_status}")
            print(f"📊 Database: {self.database_name}, Collection: {self.collection_name}")
            return True

        except AuthenticationFailure as e:
            print(f"❌ Authentication failed: {e}")
            if self.username:
                print("🔄 Retrying without authentication...")
                try:
                    # Remove credentials and retry
                    base_connection = self.connection_string.split("@")[-1]
                    self.connection_string = f"mongodb://{base_connection}"
                    self.client = MongoClient(
                        self.connection_string,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=5000
                    )
                    self.client.admin.command('ping')
                    self.db = self.client[self.database_name]
                    self.collection = self.db[self.collection_name]
                    self._create_indexes()
                    self.connected = True
                    print("✅ Connected to MongoDB without authentication")
                    return True
                except Exception as retry_error:
                    print(f"❌ Failed to connect without authentication: {retry_error}")
                    return False
            return False

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Make sure MongoDB is running")
            return False

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Create indexes for common queries
            self.collection.create_index("user_id")
            self.collection.create_index("session_id")
            self.collection.create_index([("user_id", 1), ("session_id", 1)])
            self.collection.create_index([("timestamp", -1)])  # For sorting by time
            print("✅ Created database indexes")
        except Exception as e:
            print(f"⚠️ Warning: Failed to create indexes: {e}")

    def _check_connection(self) -> bool:
        """Check if connected to MongoDB"""
        # FIX: Changed from 'not self.collection' to 'self.collection is None'
        # to avoid PyMongo 4.0+ NotImplementedError
        if not self.connected or self.collection is None:
            print("❌ Not connected to MongoDB. Please call connect() first.")
            return False
        return True

    def insert(self, user_id: str, session_id: str, question: str, answer: str) -> Optional[str]:
        """Insert a new record into the chat collection"""
        if not self._check_connection():
            return None

        try:
            # Generate unique record ID
            record_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Create record data
            record_data = {
                "_id": record_id,
                "user_id": user_id,
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "timestamp": current_time,
                "created_at": current_time,
                "updated_at": current_time
            }

            # Insert the record
            result = self.collection.insert_one(record_data)

            if result.inserted_id:
                print(f"✅ Inserted record with ID: {record_id}")
                return record_id
            return None

        except Exception as e:
            print(f"❌ Failed to insert record: {e}")
            return None

    def fetch(self, user_id: str = None, session_id: str = None,
              limit: int = None) -> List[Dict]:
        """Fetch data from MongoDB"""
        if not self._check_connection():
            return []

        try:
            # Build query filter
            query_filter = {}
            if user_id and session_id:
                query_filter = {"user_id": user_id, "session_id": session_id}
            elif user_id:
                query_filter = {"user_id": user_id}
            elif session_id:
                query_filter = {"session_id": session_id}

            # Execute query with sorting
            cursor = self.collection.find(query_filter).sort("timestamp", -1)

            # Apply limit if specified
            if limit and limit > 0:
                cursor = cursor.limit(limit)

            # Convert cursor to list
            records = list(cursor)

            # Convert ObjectId and datetime to string for JSON serialization
            for record in records:
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp'].isoformat()
                if 'created_at' in record:
                    record['created_at'] = record['created_at'].isoformat()
                if 'updated_at' in record:
                    record['updated_at'] = record['updated_at'].isoformat()

            print(f"✅ Retrieved {len(records)} records")
            return records

        except Exception as e:
            print(f"❌ Failed to fetch records: {e}")
            return []

    def fetch_qa_pairs(self, user_id: str = None, session_id: str = None,
                       limit: int = None) -> List[Dict]:
        """Fetch only question-answer pairs"""
        records = self.fetch(user_id=user_id, session_id=session_id, limit=limit)
        qa_pairs = []

        for record in records:
            question = record.get('question')
            answer = record.get('answer')
            if question and answer:
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })

        return qa_pairs

    def select_by_user(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Select all records for a specific user"""
        return self.fetch(user_id=user_id, limit=limit)

    def select_by_session(self, session_id: str, user_id: Optional[str] = None) -> List[Dict]:
        """Select records by session_id (with optional user_id filter)"""
        records = self.fetch(user_id=user_id, session_id=session_id)
        return self.extract_qa_as_list(records)

    def extract_qa_as_list(self, data_list: List[Dict]) -> List[Dict]:
        """Extract question-answer pairs from a list of dictionaries"""
        qa_list = []
        for item in data_list:
            question = item.get('question')
            answer = item.get('answer')
            if question and answer:
                qa_pair = {
                    'question': question,
                    'answer': answer
                }
                qa_list.append(qa_pair)
        return qa_list

    def update_record(self, record_id: str, **kwargs) -> bool:
        """Update a specific record"""
        if not self._check_connection():
            return False

        try:
            # Add updated_at timestamp
            kwargs['updated_at'] = datetime.now()

            # Update the record
            result = self.collection.update_one(
                {"_id": record_id},
                {"$set": kwargs}
            )

            if result.modified_count > 0:
                print(f"✅ Updated record: {record_id}")
                return True
            elif result.matched_count > 0:
                print(f"⚠️ Record {record_id} found but no changes made")
                return True
            else:
                print(f"❌ Record {record_id} not found")
                return False

        except Exception as e:
            print(f"❌ Failed to update record: {e}")
            return False

    def delete_record(self, record_id: str) -> bool:
        """Delete a specific record"""
        if not self._check_connection():
            return False

        try:
            result = self.collection.delete_one({"_id": record_id})

            if result.deleted_count > 0:
                print(f"✅ Deleted record: {record_id}")
                return True
            else:
                print(f"❌ Record {record_id} not found")
                return False

        except Exception as e:
            print(f"❌ Failed to delete record: {e}")
            return False

    def get_table_stats(self) -> Dict:
        """Get collection statistics"""
        if not self._check_connection():
            return {}

        try:
            total_records = self.collection.count_documents({})
            
            # Get unique users and sessions using aggregation
            unique_users = len(self.collection.distinct("user_id"))
            unique_sessions = len(self.collection.distinct("session_id"))

            stats = {
                "database_name": self.database_name,
                "collection_name": self.collection_name,
                "total_records": total_records,
                "unique_users": unique_users,
                "unique_sessions": unique_sessions,
                "last_updated": datetime.now().isoformat()
            }

            print("📊 Collection Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

            return stats

        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {}

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            print("✅ MongoDB connection closed")
