"""
setup_kaggle.py — Configure Kaggle API credentials.
Called once at startup before any dataset downloads.
"""
import os
import json
from pathlib import Path
import config


def setup_kaggle():
    """Write Kaggle credentials to ~/.kaggle/kaggle.json and set env vars."""
    kaggle_home = Path.home() / '.kaggle'
    kaggle_home.mkdir(exist_ok=True)
    cred_file   = kaggle_home / 'kaggle.json'

    creds = {"username": config.KAGGLE_USERNAME, "key": config.KAGGLE_KEY}
    with open(cred_file, 'w') as f:
        json.dump(creds, f)
    os.chmod(str(cred_file), 0o600)

    os.environ['KAGGLE_USERNAME']  = config.KAGGLE_USERNAME
    os.environ['KAGGLE_KEY']       = config.KAGGLE_KEY
    os.environ['KAGGLE_API_TOKEN'] = config.KAGGLE_KEY

    import kaggle
    kaggle.KaggleApi().authenticate()
    print("✅ Kaggle API authenticated.")
