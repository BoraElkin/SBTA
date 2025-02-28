import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import hashlib

class DataCache:
    """Handle caching of API responses and data"""
    
    def __init__(self, cache_dir="cache/"):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self._ensure_cache_dir()
        self._load_metadata()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self):
        """Load cache metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}
            self._save_metadata()

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_cache_key(self, source, params):
        """Generate a unique cache key based on data source and parameters"""
        param_str = json.dumps(params, sort_keys=True)
        key = f"{source}_{param_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, source, params, max_age_hours=24):
        """
        Retrieve data from cache if available and not expired
        
        Args:
            source (str): Data source identifier (e.g., 'nasa', 'noaa')
            params (dict): Request parameters
            max_age_hours (int): Maximum age of cached data in hours
            
        Returns:
            pandas.DataFrame or None: Cached data if available and valid
        """
        cache_key = self._generate_cache_key(source, params)
        
        if cache_key not in self.metadata:
            return None
            
        cache_info = self.metadata[cache_key]
        cache_file = os.path.join(self.cache_dir, cache_info['filename'])
        
        # Check if cache exists and is not expired
        if not os.path.exists(cache_file):
            del self.metadata[cache_key]
            self._save_metadata()
            return None
            
        cache_time = datetime.fromisoformat(cache_info['timestamp'])
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            logging.info(f"Cache expired for {source}")
            return None
            
        try:
            df = pd.read_parquet(cache_file)
            logging.info(f"Retrieved {source} data from cache")
            return df
        except Exception as e:
            logging.error(f"Error reading cache file: {e}")
            return None

    def save(self, df, source, params):
        """
        Save data to cache
        
        Args:
            df (pandas.DataFrame): Data to cache
            source (str): Data source identifier
            params (dict): Request parameters
        """
        if df.empty:
            logging.warning(f"Attempted to cache empty DataFrame for {source}")
            return
            
        cache_key = self._generate_cache_key(source, params)
        filename = f"{cache_key}.parquet"
        cache_file = os.path.join(self.cache_dir, filename)
        
        try:
            df.to_parquet(cache_file)
            self.metadata[cache_key] = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'params': params
            }
            self._save_metadata()
            logging.info(f"Cached {source} data successfully")
        except Exception as e:
            logging.error(f"Error caching data: {e}")

    def clear(self, source=None, max_age_hours=None):
        """
        Clear cache entries
        
        Args:
            source (str, optional): Clear only specific source
            max_age_hours (int, optional): Clear entries older than this
        """
        keys_to_remove = []
        
        for key, info in self.metadata.items():
            should_remove = False
            
            if source and info['source'] == source:
                should_remove = True
            elif max_age_hours:
                cache_time = datetime.fromisoformat(info['timestamp'])
                if datetime.now() - cache_time > timedelta(hours=max_age_hours):
                    should_remove = True
            
            if should_remove:
                cache_file = os.path.join(self.cache_dir, info['filename'])
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        if keys_to_remove:
            self._save_metadata()
            logging.info(f"Cleared {len(keys_to_remove)} cache entries") 