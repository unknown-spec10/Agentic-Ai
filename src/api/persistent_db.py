"""
Persistent database for API responses to ensure deterministic outputs.
This module provides functions to store and retrieve API responses 
in a permanent database without expiration.
"""

import os
import json
import logging
import hashlib
import sqlite3
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

# Configure logging
# Logging config handled in general_utils.py
logger = logging.getLogger('persistent_db')

# Database location
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "api_responses.db")

# Ensure database directory exists
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

class PersistentDB:
    """Manages permanent storage of API responses for deterministic outputs."""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the persistent database."""
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database and create necessary tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create responses table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                hash_key TEXT PRIMARY KEY,
                source TEXT,
                request TEXT,
                response TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON responses (source)')
            
            conn.commit()
            conn.close()
            logger.info(f"Initialized persistent database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _generate_hash_key(self, source: str, request_data: Any) -> str:
        """Generate a unique hash key for the request."""
        # Convert request_data to a consistent string representation
        if isinstance(request_data, dict):
            # Sort dictionary to ensure consistent ordering
            request_str = json.dumps(request_data, sort_keys=True)
        else:
            request_str = str(request_data)
        
        # Combine source and request for a unique hash
        combined = f"{source}:{request_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def store_response(self, source: str, request_data: Any, response_data: Any, 
                      metadata: Optional[Dict] = None) -> str:
        """
        Store an API response permanently in the database.
        
        Args:
            source: The API or data source identifier (e.g., 'gemini', 'github')
            request_data: The original request parameters
            response_data: The response data to store
            metadata: Optional metadata about the response
            
        Returns:
            The hash key used to store the response
        """
        hash_key = self._generate_hash_key(source, request_data)
        
        try:
            # Convert data to JSON strings
            request_json = json.dumps(request_data)
            response_json = json.dumps(response_data)
            metadata_json = json.dumps(metadata or {})
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we already have this response
            cursor.execute('SELECT hash_key FROM responses WHERE hash_key = ?', (hash_key,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                UPDATE responses
                SET response = ?, metadata = ?
                WHERE hash_key = ?
                ''', (response_json, metadata_json, hash_key))
                logger.info(f"Updated existing response for {source} request")
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO responses (hash_key, source, request, response, metadata)
                VALUES (?, ?, ?, ?, ?)
                ''', (hash_key, source, request_json, response_json, metadata_json))
                logger.info(f"Stored new response for {source} request")
            
            conn.commit()
            conn.close()
            return hash_key
        
        except Exception as e:
            logger.error(f"Error storing response: {e}")
            return hash_key
    
    def get_response(self, source: str, request_data: Any) -> Optional[Any]:
        """
        Retrieve a stored API response from the database.
        
        Args:
            source: The API or data source identifier
            request_data: The original request parameters
            
        Returns:
            The stored response data or None if not found
        """
        hash_key = self._generate_hash_key(source, request_data)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT response FROM responses WHERE hash_key = ?', (hash_key,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                logger.info(f"Retrieved stored response for {source} request")
                return json.loads(result[0])
            else:
                logger.info(f"No stored response found for {source} request")
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving response: {e}")
            return None
    
    def response_exists(self, source: str, request_data: Any) -> bool:
        """Check if a response exists for the given source and request."""
        hash_key = self._generate_hash_key(source, request_data)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT 1 FROM responses WHERE hash_key = ?', (hash_key,))
            result = cursor.fetchone()
            
            conn.close()
            return result is not None
        
        except Exception as e:
            logger.error(f"Error checking if response exists: {e}")
            return False
    
    def delete_response(self, source: str, request_data: Any) -> bool:
        """Delete a stored API response from the database."""
        hash_key = self._generate_hash_key(source, request_data)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM responses WHERE hash_key = ?', (hash_key,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted stored response for {source} request")
            
            return deleted
        
        except Exception as e:
            logger.error(f"Error deleting response: {e}")
            return False
    
    def get_all_sources(self) -> List[str]:
        """Get a list of all distinct sources in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT source FROM responses')
            sources = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return sources
        
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []

# Singleton instance
_persistent_db = None

def get_persistent_db() -> PersistentDB:
    """Get the singleton PersistentDB instance."""
    global _persistent_db
    if _persistent_db is None:
        _persistent_db = PersistentDB()
    return _persistent_db

# Helper functions for common database operations

def store_api_response(source: str, request_data: Any, response_data: Any, 
                     metadata: Optional[Dict] = None) -> str:
    """Store an API response in the persistent database."""
    db = get_persistent_db()
    return db.store_response(source, request_data, response_data, metadata)

def get_api_response(source: str, request_data: Any) -> Optional[Any]:
    """Get a stored API response from the persistent database."""
    db = get_persistent_db()
    return db.get_response(source, request_data)

def api_response_exists(source: str, request_data: Any) -> bool:
    """Check if an API response exists in the persistent database."""
    db = get_persistent_db()
    return db.response_exists(source, request_data)

def delete_api_response(source: str, request_data: Any) -> bool:
    """Delete an API response from the persistent database."""
    db = get_persistent_db()
    return db.delete_response(source, request_data)
