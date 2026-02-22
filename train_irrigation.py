"""
train_irrigation.py — Module 3: Paddy Irrigation Prediction
Binary classification: does paddy need irrigation? (0=No, 1=Yes)

Run:  python train_irrigation.py
Skip: Automatically skipped if saved_models/irrigation/model.pkl already exists.
"""

import os
import json
import warnings
import shutil
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                               classification_report)
import xgboost as xgb
import lightgbm as lgb

import config

warnings.filterwarnings('ignore')


# ──────────────────────── Synthetic augmentation ──────────────────────────────

def _generate_synthetic(n_samples=600, random_state=42):
    """Generate agronomically-sound synthetic paddy irrigation records."""
    rng           = np.random.RandomState(random_state)
    crop_days     = rng.randint(1, 121, n_samples)
    soil_moisture = rng.randint(100, 801, n_samples).astype(float)
    temperature   = rng.uniform(20, 42, n_samples)
    humidity      = rng.uniform(10, 70, n_samples)
    irrigation    = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        sm, temp, hum, days = soil_moisture[i], temperature[i], humidity[i], crop_days[i]
        score = 0.0
        if sm < 200:    score += 3.0
        elif sm < 300:  score += 2.0
        elif sm < 400:  score += 1.0
        elif sm < 500:  score += 0.2
        else:           score -= 1.0
        if temp > 36:   score += 1.5
        elif temp > 32: score += 1.0
        elif temp > 28: score += 0.3
        else:           score -= 0.3
        if hum < 20:    score += 1.2
        elif hum < 30:  score += 0.7
        elif hum < 40:  score += 0.2
        elif hum > 60:  score -= 0.5
        if days <= 20:   score += 0.8
        elif days <= 60: score += 0.4
        elif days >= 90: score -= 0.8
        score += rng.normal(0, 0.4)
        irrigation[i] = 1 if score >= 1.5 else 0

    return pd.DataFrame({
        'CropDays'    : crop_days,
        'SoilMoisture': soil_moisture.astype(int),
        'temperature' : np.round(temperature).astype(int),
        'Humidity'    : np.round(humidity).astype(int),
        'Irrigation'  : irrigation,
    })


# ──────────────────────── Download ───────────────────────────────────────────

def _download_dataset():
    if Path(config.IRR_CSV).exists():
        print(f'  ✅ Irrigation dataset already exists.')
        return
    print('  📥 Downloading irrigation dataset...')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files('pusainstitute/cropirrigationscheduling',
                                path=config.IRR_DATA_DIR, unzip=True)
    # Rename to canonical path
    csv_files = list(Path(config.IRR_DATA_DIR).glob('*.csv'))
    target    = Path(config.IRR_CSV)
    if csv_files and csv_files[0] != target:
        shutil.move(str(csv_files[0]), str(target))
    print('  ✅ Irrigation dataset downloaded!')


# ──────────────────────── Main ───────────────────────────────────────────────

def train():
    """Full irrigation training pipeline. Skipped if model already saved."""
    os.makedirs(config.MODEL_DIR_IRR, exist_ok=True)

    if os.path.exists(config.IRR_MODEL_PATH):
        print('⏭️  Irrigation model already trained — skipping. Delete saved_models/irrigation/ to retrain.')
        return

    print('\n' + '='*60)
    print('💧  MODULE 3 — Paddy Irrigation Prediction')
    print('='*60)

    # 1. Download
    Path(config.IRR_DATA_DIR).mkdir(exist_ok=True)
    _download_dataset()

    # 2. Load & filter paddy rows
    df_full = pd.read_csv(config.IRR_CSV)
    df_raw  = df_full[df_full['CropType'] == 'Paddy'].copy()
    df_raw  = df_raw.drop('CropType', axis=1).reset_index(drop=True)
    print(f'  Paddy records: {len(df_raw)} | Balance: {df_raw["Irrigation"].value_counts().to_dict()}')

    # 3. Augment with synthetic data
    df_synth = _generate_synthetic(550, 42)
    df_model = pd.concat([df_raw, df_synth], ignore_index=True)
    print(f'  Augmented: {df_model.shape} | Balance: {df_model["Irrigation"].value_counts().to_dict()}')

    # Drop extra cols that came from synthetic but not raw and vice versa
    df_model = df_model[['CropDays','SoilMoisture','temperature','Humidity','Irrigation']].copy()

    # 4. Prepare features
    X = df_model.drop('Irrigation', axis=1)
    y = df_model['Irrigation']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scale_needed = {'Logistic Regression', 'SVM'}

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM'                : SVC(probability=True, kernel='rbf', random_state=42),
        'XGBoost'            : xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                   eval_metric='logloss', verbosity=0),
        'LightGBM'           : lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    # 5. Train & compare
    print(f'\n  {"Model":<25} {"CV Acc":>10} {"Test Acc":>10} {"F1":>8} {"AUC":>8}')
    print('  ' + '-'*65)
    results, trained = [], {}
    for name, mdl in models.items():
        Xtr = X_tr_sc if name in scale_needed else X_tr
        Xte = X_te_sc if name in scale_needed else X_te
        cv_s = cross_val_score(mdl, Xtr, y_tr, cv=cv, scoring='accuracy')
        mdl.fit(Xtr, y_tr)
        y_pred = mdl.predict(Xte)
        y_prob = mdl.predict_proba(Xte)[:, 1]
        results.append({
            'Model': name, 'CV Accuracy': cv_s.mean(), 'CV Std': cv_s.std(),
            'Test Accuracy': accuracy_score(y_te, y_pred),
            'F1-Score': f1_score(y_te, y_pred),
            'AUC': roc_auc_score(y_te, y_prob),
            'needs_scale': name in scale_needed,
        })
        trained[name] = mdl
        print(f'  {name:<25} {cv_s.mean():>10.4f} {accuracy_score(y_te, y_pred):>10.4f} '
              f'{f1_score(y_te, y_pred):>8.4f} {roc_auc_score(y_te, y_prob):>8.4f}')

    cmp_df = pd.DataFrame(results)

    # 6. Tune best with GridSearch (Random Forest)
    print('\n  🔧 Tuning Random Forest...')
    best_row = cmp_df.loc[cmp_df['F1-Score'].idxmax()]
    rf_grid  = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 8, 12, None],
         'min_samples_split': [2, 5, 10], 'class_weight': ['balanced', None]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_tr, y_tr)
    best_rf       = rf_grid.best_estimator_
    y_rf_pred     = best_rf.predict(X_te)
    tuned_f1      = f1_score(y_te, y_rf_pred)

    if tuned_f1 >= best_row['F1-Score']:
        final_model      = best_rf
        final_model_name = 'Random Forest (Tuned)'
        needs_scale      = False
    else:
        best_name        = best_row['Model']
        final_model      = trained[best_name]
        final_model_name = best_name
        needs_scale      = bool(best_row['needs_scale'])

    X_te_final   = X_te if not needs_scale else X_te_sc
    y_pred_final = final_model.predict(X_te_final)
    y_prob_final = final_model.predict_proba(X_te_final)[:, 1]
    test_acc     = accuracy_score(y_te, y_pred_final)
    test_f1      = f1_score(y_te, y_pred_final)
    test_auc     = roc_auc_score(y_te, y_prob_final)

    print(f'\n  🏆 Final model: {final_model_name}')
    print(f'  Acc: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}')

    # 7. Save
    joblib.dump(final_model, config.IRR_MODEL_PATH)
    joblib.dump(scaler,      config.IRR_SCALER_PATH)

    cv_final = cross_val_score(
        final_model,
        X if not needs_scale else scaler.transform(X),
        y, cv=cv, scoring='accuracy'
    )
    metadata = {
        'model_name'         : final_model_name,
        'test_accuracy'      : float(round(test_acc, 4)),
        'cv_accuracy_mean'   : float(round(cv_final.mean(), 4)),
        'cv_accuracy_std'    : float(round(cv_final.std(), 4)),
        'f1_score'           : float(round(test_f1, 4)),
        'auc'                : float(round(test_auc, 4)),
        'total_paddy_samples': int(len(df_model)),
        'needs_scale'        : bool(needs_scale),
        'feature_columns'    : X.columns.tolist(),
    }
    with open(config.IRR_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  💾 Irrigation model saved → {config.IRR_MODEL_PATH}')
    print(f'  ✅ Done! ({final_model_name}  Acc={test_acc:.4f})')


if __name__ == '__main__':
    from setup_kaggle import setup_kaggle
    setup_kaggle()
    train()
