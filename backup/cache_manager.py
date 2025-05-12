"""
Cache management utilities for storing and retrieving API responses and skill data.
Implements a file-based cache system with TTL (Time To Live) functionality.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import threading
import hashlib

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Cache configuration
DEFAULT_TTL = 60 * 60 * 24 * 7  # 7 days in seconds
CACHE_REFRESH_INTERVAL = 60 * 60 * 24  # 1 day in seconds

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cache_manager')

class CacheManager:
    """Manages caching of API responses and skill data with TTL."""
    
    def __init__(self, cache_dir: str = CACHE_DIR, default_ttl: int = DEFAULT_TTL):
        """Initialize the cache manager with directory and TTL."""
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.refresh_thread = None
        self.stop_refresh = False
        
        # Start background refresh thread
        self._start_refresh_thread()
    
    def _start_refresh_thread(self):
        """Start a background thread to periodically refresh cache items."""
        if self.refresh_thread is None or not self.refresh_thread.is_alive():
            self.stop_refresh = False
            self.refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
            self.refresh_thread.start()
            logger.info("Started cache refresh background thread")
    
    def _background_refresh(self):
        """Background thread that periodically refreshes cache items."""
        while not self.stop_refresh:
            try:
                self._refresh_expiring_items()
                time.sleep(CACHE_REFRESH_INTERVAL)
            except Exception as e:
                logger.error(f"Error in background refresh: {e}")
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _refresh_expiring_items(self):
        """Refresh cache items that will expire soon."""
        logger.info("Checking for expiring cache items...")
        try:
            for cache_file in os.listdir(self.cache_dir):
                if not cache_file.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, cache_file)
                try:
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check if it's a cache item with metadata
                    if not isinstance(cache_data, dict) or 'expires_at' not in cache_data:
                        continue
                    
                    expires_at = cache_data['expires_at']
                    now = time.time()
                    
                    # If it will expire in the next 24 hours, try to refresh it
                    if 0 < expires_at - now < CACHE_REFRESH_INTERVAL:
                        refresh_func = cache_data.get('refresh_func')
                        if refresh_func:
                            logger.info(f"Refreshing cache item: {cache_file}")
                            # Here you would call the refresh function
                            # This requires integration with your data sources
                        else:
                            logger.info(f"No refresh function for cache item: {cache_file}")
                
                except Exception as e:
                    logger.error(f"Error processing cache file {cache_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error refreshing cache items: {e}")
    
    def stop(self):
        """Stop the background refresh thread."""
        if self.refresh_thread and self.refresh_thread.is_alive():
            self.stop_refresh = True
            self.refresh_thread.join(timeout=2.0)
            logger.info("Stopped cache refresh thread")
    
    def _get_cache_file_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Create a safe filename from the key using hash
        safe_filename = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_filename}.json")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache, return default if not found or expired."""
        file_path = self._get_cache_file_path(key)
        
        if not os.path.exists(file_path):
            return default
        
        try:
            with open(file_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if it's a cache item with metadata
            if not isinstance(cache_data, dict) or 'data' not in cache_data:
                return cache_data  # Return raw data for backwards compatibility
            
            # Check expiration if it exists
            expires_at = cache_data.get('expires_at')
            if expires_at and time.time() > expires_at:
                logger.info(f"Cache expired for key: {key}")
                return default
            
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Error reading cache for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, refresh_func: Optional[str] = None) -> bool:
        """Set a value in the cache with optional TTL and refresh function."""
        file_path = self._get_cache_file_path(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        try:
            cache_data = {
                'data': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl,
                'refresh_func': refresh_func
            }
            
            with open(file_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cache set for key: {key}, TTL: {ttl} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        file_path = self._get_cache_file_path(key)
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            logger.info(f"Cache deleted for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            for cache_file in os.listdir(self.cache_dir):
                if cache_file.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, cache_file)
                    os.remove(file_path)
            
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the singleton CacheManager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

# Helper functions for common cache operations

def get_cached(key: str, default: Any = None) -> Any:
    """Get a value from the cache."""
    return get_cache_manager().get(key, default)

def set_cached(key: str, value: Any, ttl: Optional[int] = None, refresh_func: Optional[str] = None) -> bool:
    """Set a value in the cache."""
    return get_cache_manager().set(key, value, ttl, refresh_func)

def delete_cached(key: str) -> bool:
    """Delete a value from the cache."""
    return get_cache_manager().delete(key)

def clear_cache() -> bool:
    """Clear all cache entries."""
    return get_cache_manager().clear()
