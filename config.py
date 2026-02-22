"""
config.py — Central configuration for Rice AI Suite
All paths, seeds, and constants live here.
"""
import os
import random
import platform
import numpy as np
import torch

# ── Seeds ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SEED         = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Runtime compatibility toggles ───────────────────────────────────────────
IS_MACOS = platform.system() == 'Darwin'
ENABLE_LIGHTGBM = os.environ.get('ENABLE_LIGHTGBM', '0' if IS_MACOS else '1') == '1'
DISEASE_USE_PRETRAINED = os.environ.get(
    'DISEASE_USE_PRETRAINED',
    '0' if (IS_MACOS and DEVICE == 'cpu') else '1'
) == '1'
DISEASE_DATALOADER_WORKERS = int(os.environ.get('DISEASE_DATALOADER_WORKERS', '0' if IS_MACOS else '2'))
DISEASE_PIN_MEMORY = os.environ.get('DISEASE_PIN_MEMORY', '1' if DEVICE == 'cuda' else '0') == '1'
DISEASE_PROGRESS_EVERY = int(os.environ.get('DISEASE_PROGRESS_EVERY', '10'))
DISEASE_FAST_MODE = os.environ.get('DISEASE_FAST_MODE', '1' if DEVICE == 'cpu' else '0') == '1'

# ── Paths — Data ─────────────────────────────────────────────────────────────
YIELD_DATA_DIR    = './datasets/yield'
YIELD_CSV         = f'{YIELD_DATA_DIR}/paddydataset.csv'

DISEASE_DATA_DIR  = './datasets/disease'
DISEASE_TRAIN_DIR = f'{DISEASE_DATA_DIR}/train_images'
DISEASE_TEST_DIR  = f'{DISEASE_DATA_DIR}/test_images'
DISEASE_CSV       = f'{DISEASE_DATA_DIR}/train.csv'

IRR_DATA_DIR      = './datasets/irrigation'
IRR_CSV           = f'{IRR_DATA_DIR}/datasets_-_datasets.csv'

FERT_DATA_DIR     = './datasets/fertilizer'

# ── Paths — Models ────────────────────────────────────────────────────────────
MODEL_DIR_YIELD      = './models/yield'
MODEL_DIR_DISEASE    = './models/disease'
MODEL_DIR_IRR        = './models/irrigation'
MODEL_DIR_FERT       = './models/fertilizer'

YIELD_MODEL_PATH     = f'{MODEL_DIR_YIELD}/model.pkl'
YIELD_SCALER_PATH    = f'{MODEL_DIR_YIELD}/scaler.pkl'
YIELD_ENCODERS_PATH  = f'{MODEL_DIR_YIELD}/label_encoders.pkl'
YIELD_FEATURES_PATH  = f'{MODEL_DIR_YIELD}/features.pkl'
YIELD_SUMMARY_PATH   = f'{MODEL_DIR_YIELD}/summary.json'

DISEASE_MODEL_PATH   = f'{MODEL_DIR_DISEASE}/best.pth'
DISEASE_ENCODER_PATH = f'{MODEL_DIR_DISEASE}/label_encoder.pkl'
DISEASE_CONFIG_PATH  = f'{MODEL_DIR_DISEASE}/config.json'

IRR_MODEL_PATH       = f'{MODEL_DIR_IRR}/model.pkl'
IRR_SCALER_PATH      = f'{MODEL_DIR_IRR}/scaler.pkl'
IRR_METADATA_PATH    = f'{MODEL_DIR_IRR}/metadata.json'

FERT_MODEL_PATH      = f'{MODEL_DIR_FERT}/model.pkl'
FERT_SCALER_PATH     = f'{MODEL_DIR_FERT}/scaler.pkl'
FERT_ENCODER_PATH    = f'{MODEL_DIR_FERT}/label_encoder.pkl'
FERT_FEATURES_PATH   = f'{MODEL_DIR_FERT}/feature_names.pkl'

# ── Yield feature names ───────────────────────────────────────────────────────
YIELD_TARGET = 'Paddy yield(in Kg)'
YIELD_SEL7   = [
    'LP_nurseryarea(in Tonnes)', 'DAP_20days', 'Urea_40Days',
    'Pest_60Day(in ml)', 'Seedrate(in Kg)', 'Hectares', '30DRain( in mm)',
]
YIELD_ENG      = ['Fertilizer_per_Ha', 'Input_Intensity', 'Seed_Density', 'Rain_per_Ha']
YIELD_FEATURES = YIELD_SEL7 + YIELD_ENG

# ── Disease — image transforms ────────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
IMG1, BS1, EPOCHS1, LR1 = 224, 64, 6, 3e-4   # Stage 1
IMG2, BS2, EPOCHS2, LR2 = 320, 64, 3, 5e-5   # Stage 2

# ── API Credentials (REQUIRED) ───────────────────────────────────────────────
# All credentials MUST be set via environment variables or .env file
# See .env.example for template
# Get Kaggle API key: https://www.kaggle.com/settings/account
# Get Gemini API key: https://makersuite.google.com/app/apikey

class _LazyCredential:
    """Lazy credential loader - only validates when first accessed."""
    def __init__(self, key: str, description: str, required: bool = True):
        self.key = key
        self.description = description
        self.required = required
        self._value = None
        self._loaded = False
    
    def __str__(self) -> str:
        """Access credential value (validates on first access if required)."""
        if not self._loaded:
            self._value = os.environ.get(self.key)
            if not self._value and self.required:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"❌ MISSING REQUIRED CREDENTIAL: {self.key}\n"
                    f"   {self.description}\n\n"
                    f"   Setup Instructions:\n"
                    f"   1. Copy .env.example to .env\n"
                    f"   2. Fill in your {self.key}\n"
                    f"   3. Load environment: export $(cat .env | xargs)\n\n"
                    f"   Or set directly: export {self.key}='your_value'\n"
                    f"{'='*70}"
                )
            self._loaded = True
        return self._value if self._value else ''
    
    def __bool__(self) -> bool:
        """Check if credential is set."""
        return bool(str(self))

# Lazy credential loading - only validates when accessed
KAGGLE_USERNAME = _LazyCredential('KAGGLE_USERNAME', 'Your Kaggle username', required=True)
KAGGLE_KEY      = _LazyCredential('KAGGLE_KEY', 'Your Kaggle API key', required=True)
GEMINI_API_KEY  = _LazyCredential('GEMINI_API_KEY', 'Your Google Gemini API key (optional for AI tips)', required=False)
