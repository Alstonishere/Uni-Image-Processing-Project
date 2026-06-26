import json
import os
import hashlib

USER_DB = "users.json"

def get_all_usernames():
    if not os.path.exists(USER_DB):
        return[]
    
    try:
        with open(USER_DB, "r") as f:
            data = f.read().strip()
            users = json.loads(data) if data else {}

            return list(users.keys())
        
    except(json.JSONDecodeError,ValueError):
        return[]
    
    def generate_session_key(sender, recipient, pin):
        # Combine inputs into a single string        
        raw_data = f"{sender}-{recipient}-{pin}"
        
        # Create SHA-256 hash
        key_hash = hashlib.sha256(raw_data.encode()).hexdigest()
        
        # Convert the first 8 characters of the hash to an integer 
        # to serve as a numeric seed for random/chaos functions
        seed = int(key_hash[:8], 16)
        
        return seed, key_hash