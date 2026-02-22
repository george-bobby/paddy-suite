"""
load_env.py — Load environment variables from .env file
Run this before main.py if you're not using shell export
"""
import os
from pathlib import Path


def load_env_file(env_path='.env'):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"⚠️  No .env file found at {env_file.absolute()}")
        print(f"   Please copy .env.example to .env and fill in your credentials")
        print(f"   Command: cp .env.example .env")
        return False
    
    loaded_count = 0
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Only set if not already in environment and not a placeholder
                if value and not value.startswith('your_'):
                    os.environ[key] = value
                    loaded_count += 1
    
    if loaded_count > 0:
        print(f"✅ Loaded {loaded_count} environment variables from .env")
        return True
    else:
        print(f"⚠️  No valid credentials found in .env file")
        print(f"   Please edit .env and replace placeholder values with your actual credentials")
        return False


if __name__ == '__main__':
    load_env_file()
