"""
config.py — Central configuration for Rice AI Suite
All paths, seeds, and constants live here.
"""
import os
import random
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

# ── Paths — Data ─────────────────────────────────────────────────────────────
YIELD_DATA_DIR    = './crop_data'
YIELD_CSV         = f'{YIELD_DATA_DIR}/paddydataset.csv'

DISEASE_DATA_DIR  = './paddy_data'
DISEASE_TRAIN_DIR = f'{DISEASE_DATA_DIR}/train_images'
DISEASE_TEST_DIR  = f'{DISEASE_DATA_DIR}/test_images'
DISEASE_CSV       = f'{DISEASE_DATA_DIR}/train.csv'

IRR_DATA_DIR      = './data'
IRR_CSV           = f'{IRR_DATA_DIR}/datasets_-_datasets.csv'

FERT_DATA_DIR     = './rice_fertilizer_data'

# ── Paths — Models ────────────────────────────────────────────────────────────
MODEL_DIR_YIELD      = './saved_models/yield'
MODEL_DIR_DISEASE    = './saved_models/disease'
MODEL_DIR_IRR        = './saved_models/irrigation'
MODEL_DIR_FERT       = './saved_models/fertilizer'

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

# ── Kaggle credentials — EDIT these or set env vars KAGGLE_USERNAME / KAGGLE_KEY
KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME', 'deonasaji')
KAGGLE_KEY      = os.environ.get('KAGGLE_KEY',      'KGAT_e48eb4132db7726c40438655aa2c2cf4')

# ── Gemini API key — EDIT or set env var GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDr3j6meyCxNCHbYZqnnbwxMrv8BFZ48xc')
