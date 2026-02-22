"""
train_fertilizer.py — Module 4: Rice Fertilizer Recommendation
Multi-class classification with realistic noise injection.

Run:  python train_fertilizer.py
Skip: Automatically skipped if models/fertilizer/model.pkl already exists.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV)
from sklearn.metrics import accuracy_score, f1_score, classification_report

import config

warnings.filterwarnings('ignore')


# ──────────────────────── Download ───────────────────────────────────────────

def _download_dataset():
    data_dir = Path(config.FERT_DATA_DIR)
    if list(data_dir.glob('*.csv')):
        print(f'  ✅ Fertilizer dataset already exists.')
        return
    print('  📥 Downloading fertilizer dataset...')
    import kaggle
    kaggle.api.dataset_download_files(
        'hamzmiscof/rice-fertilizer-recommendation-dataset',
        path=config.FERT_DATA_DIR, unzip=True
    )
    print('  ✅ Fertilizer dataset downloaded!')


def _get_fert_csv():
    return str(list(Path(config.FERT_DATA_DIR).glob('*.csv'))[0])


# ──────────────────────── Noise injection ─────────────────────────────────────

def _inject_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic noise to break perfect synthetic correlations."""
    df = df.copy()
    noise_params = {'moist': 5.0, 'soilT': 1.5, 'EC': 150.0, 'airT': 1.0, 'airH': 3.0}
    for feat, std in noise_params.items():
        df[feat] += np.random.normal(0, std, len(df))
    df['moist']            = df['moist'].clip(20, 100)
    df[['soilT', 'airT']] = df[['soilT', 'airT']].clip(20, 40)
    df['airH']             = df['airH'].clip(40, 100)
    df['EC']               = df['EC'].clip(0, 3500)

    # Break Fase_Tanam correlation (flip 20%)
    flip_idx = np.random.choice(df.index, size=int(len(df) * 0.20), replace=False)
    df.loc[flip_idx, 'Fase_Tanam'] = df.loc[flip_idx, 'Fase_Tanam'].apply(
        lambda x: 1000 if x == 0 else 0)

    # 5% label noise
    mislabel_idx = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    for idx in mislabel_idx:
        cur = df.loc[idx, 'Label']
        df.loc[idx, 'Label'] = np.random.choice([l for l in df['Label'].unique() if l != cur])

    return df


# ──────────────────────── Main ───────────────────────────────────────────────

def train():
    """Full fertilizer training pipeline. Skipped if model already saved."""
    os.makedirs(config.MODEL_DIR_FERT, exist_ok=True)

    if os.path.exists(config.FERT_MODEL_PATH):
        print('⏭️  Fertilizer model already trained — skipping. Delete models/fertilizer/ to retrain.')
        return

    print('\n' + '='*60)
    print('🌱  MODULE 4 — Rice Fertilizer Recommendation')
    print('='*60)

    # 1. Download
    Path(config.FERT_DATA_DIR).mkdir(exist_ok=True)
    _download_dataset()

    # 2. Load & inject noise
    fert_csv = _get_fert_csv()
    df_orig  = pd.read_csv(fert_csv)
    print(f'  Original shape: {df_orig.shape} | Classes: {list(df_orig["Label"].unique())}')
    df       = _inject_noise(df_orig)
    print(f'  ✅ Realistic noise added | Classes: {df["Label"].value_counts().to_dict()}')

    # 3. Preprocess
    X = df.drop('Label', axis=1)
    y = df['Label']
    fert_feature_names = X.columns.tolist()

    le         = LabelEncoder()
    y_enc      = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y_enc
    )
    scaler   = RobustScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    model_defs = {
        'Random Forest': {
            'model' : RandomForestClassifier(random_state=config.RANDOM_STATE),
            'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, 15],
                       'min_samples_split': [10, 20], 'min_samples_leaf': [5, 10],
                       'max_features': ['sqrt', 'log2']}
        },
        'Gradient Boosting': {
            'model' : GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05, 0.1],
                       'max_depth': [3, 5], 'min_samples_split': [10, 20], 'subsample': [0.8, 0.9]}
        },
        'Logistic Regression': {
            'model' : LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000),
            'params': {'C': [0.01, 0.1, 1.0], 'penalty': ['l2'], 'solver': ['lbfgs']}
        },
    }

    # 4. Train & compare
    best_f1, best_name, best_model = 0, None, None
    model_results = {}
    for name, info in model_defs.items():
        print(f'\n  Training: {name}')
        gs = GridSearchCV(info['model'], info['params'], cv=cv,
                          scoring='f1_weighted', n_jobs=-1, verbose=0)
        gs.fit(X_tr_sc, y_tr)
        est     = gs.best_estimator_
        cv_s    = cross_val_score(est, X_tr_sc, y_tr, cv=cv, scoring='f1_weighted')
        y_pred  = est.predict(X_te_sc)
        acc     = accuracy_score(y_te, y_pred)
        f1      = f1_score(y_te, y_pred, average='weighted')
        model_results[name] = {'model': est, 'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(),
                                'test_accuracy': acc, 'f1_score': f1}
        print(f'  CV F1: {cv_s.mean():.4f} ± {cv_s.std():.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}')
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, est

    print(f'\n  🏆 Best: {best_name}  (F1={best_f1:.4f})')

    # 5. Save
    joblib.dump(best_model,        config.FERT_MODEL_PATH)
    joblib.dump(scaler,            config.FERT_SCALER_PATH)
    joblib.dump(le,                config.FERT_ENCODER_PATH)
    joblib.dump(fert_feature_names, config.FERT_FEATURES_PATH)

    print(f'  💾 Fertilizer model saved → {config.FERT_MODEL_PATH}')
    print(f'  ✅ Done! ({best_name}  F1={best_f1:.4f})')


if __name__ == '__main__':
    from setup_kaggle import setup_kaggle
    setup_kaggle()
    train()
