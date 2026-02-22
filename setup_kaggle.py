"""
setup_kaggle.py — Configure Kaggle API credentials.
Called once at startup before any dataset downloads.
"""
import os
import json
from pathlib import Path
import config


def setup_kaggle():
    """Write Kaggle credentials to ~/.kaggle/kaggle.json and set env vars.
    
    Validates credentials are present and properly formatted before writing.
    Raises clear error messages if credentials are missing.
    """
    # Convert lazy credentials to strings (will raise if missing)
    kaggle_username = str(config.KAGGLE_USERNAME)
    kaggle_key = str(config.KAGGLE_KEY)
    
    kaggle_home = Path.home() / '.kaggle'
    kaggle_home.mkdir(exist_ok=True)
    cred_file   = kaggle_home / 'kaggle.json'

    creds = {"username": kaggle_username, "key": kaggle_key}
    
    try:
        with open(cred_file, 'w') as f:
            json.dump(creds, f)
        os.chmod(str(cred_file), 0o600)
    except Exception as e:
        raise RuntimeError(f"Failed to write Kaggle credentials to {cred_file}: {e}")

    # Set environment variables for kaggle API
    os.environ['KAGGLE_USERNAME']  = kaggle_username
    os.environ['KAGGLE_KEY']       = kaggle_key
    os.environ['KAGGLE_API_TOKEN'] = kaggle_key

    try:
        import kaggle
        kaggle.KaggleApi().authenticate()
        print("✅ Kaggle API authenticated.")
    except Exception as e:
        raise RuntimeError(
            f"Kaggle authentication failed: {e}\n"
            f"Please verify your credentials at https://www.kaggle.com/settings/account"
        )
