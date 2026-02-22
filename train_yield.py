"""
train_yield.py — Module 1: Paddy Crop Yield Prediction
Trains an XGBoost/LightGBM/RandomForest regressor with Optuna tuning.

Run:  python train_yield.py
Skip: Automatically skipped if models/yield/model.pkl already exists.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
try:
    import lightgbm as lgb
except Exception:
    lgb = None
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
from src.data import DatasetManager
from src.utils import ModelArtifacts

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────── helpers ─────────────────────────────────────────

def _download_dataset():
    """Download yield dataset from Kaggle using DatasetManager."""
    dataset_manager = DatasetManager()
    dataset_manager.download_kaggle_dataset(
        dataset='stealthtechnologies/predict-crop-production',
        output_dir=config.YIELD_DATA_DIR,
        expected_file='paddydataset.csv'
    )


def _feature_engineering(df):
    """Impute, encode, and add engineered features."""
    data = df.copy()
    num_cols = data.select_dtypes(include='number').columns.tolist()
    cat_cols = data.select_dtypes(include='object').columns.tolist()

    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    data['Fertilizer_per_Ha'] = data['LP_nurseryarea(in Tonnes)'] / (data['Hectares'] + 1e-5)
    data['Input_Intensity']   = (data['DAP_20days'] +
                                  data['Pest_60Day(in ml)'] / 10 +
                                  data['LP_nurseryarea(in Tonnes)'] * 100)
    data['Seed_Density']      = data['Seedrate(in Kg)'] / (data['Hectares'] + 1e-5)
    data['Rain_per_Ha']       = data['30DRain( in mm)'] / (data['Hectares'] + 1e-5)

    return data, label_encoders


def _evaluate(model, Xtr, Xte, ytr, yte, name):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    kf     = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_r2  = cross_val_score(model, Xtr, ytr, cv=kf, scoring='r2')
    return {
        'Model'     : name,
        'Test R2'   : round(r2_score(yte, y_pred), 4),
        'Test MAE'  : round(mean_absolute_error(yte, y_pred), 1),
        'Test RMSE' : round(np.sqrt(mean_squared_error(yte, y_pred)), 1),
        'CV R2 Mean': round(cv_r2.mean(), 4),
        'CV R2 Std' : round(cv_r2.std(), 4),
        '_model'    : model,
        '_preds'    : y_pred,
    }


def _optuna_objective(trial, best_name, Xtr, ytr):
    if 'XGBoost' in best_name:
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 200, 800),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            max_depth        = trial.suggest_int('max_depth', 3, 8),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        )
        mdl = xgb.XGBRegressor(**params, random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0)
    elif 'LightGBM' in best_name and lgb is not None:
        params = dict(
            n_estimators  = trial.suggest_int('n_estimators', 200, 800),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            num_leaves    = trial.suggest_int('num_leaves', 20, 100),
        )
        mdl = lgb.LGBMRegressor(**params, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
    else:
        params = dict(
            n_estimators = trial.suggest_int('n_estimators', 100, 500),
            max_depth    = trial.suggest_int('max_depth', 5, 15),
        )
        mdl = RandomForestRegressor(**params, random_state=config.RANDOM_STATE, n_jobs=-1)
    kf = KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_score(mdl, Xtr, ytr, cv=kf, scoring='r2').mean()


# ─────────────────────────── main ────────────────────────────────────────────

def train():
    """Full yield training pipeline. Skipped if model already saved."""
    os.makedirs(config.MODEL_DIR_YIELD, exist_ok=True)

    if os.path.exists(config.YIELD_MODEL_PATH):
        print('⏭️  Yield model already trained — skipping. Delete models/yield/ to retrain.')
        return

    print('\n' + '='*60)
    print('🌾  MODULE 1 — Paddy Crop Yield Prediction')
    print('='*60)

    # 1. Download
    os.makedirs(config.YIELD_DATA_DIR, exist_ok=True)
    _download_dataset()

    # 2. Load
    df = pd.read_csv(config.YIELD_CSV)
    df.columns = df.columns.str.strip()
    print(f'  Shape: {df.shape} | Missing: {df.isnull().sum().sum()}')

    # 3. Preprocess
    data, label_encoders = _feature_engineering(df)
    X = data[config.YIELD_FEATURES]
    y = data[config.YIELD_TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # 4. Compare models
    model_zoo = {
        'Ridge'         : Ridge(alpha=10),
        'Random Forest' : RandomForestRegressor(n_estimators=300, max_depth=10,
                              random_state=config.RANDOM_STATE, n_jobs=-1),
        'Grad Boosting' : GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=5, random_state=config.RANDOM_STATE),
        'XGBoost'       : xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0),
    }

    if config.ENABLE_LIGHTGBM and lgb is not None:
        model_zoo['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        print('  ℹ️ LightGBM disabled for this environment (set ENABLE_LIGHTGBM=1 to force-enable).')

    print(f'\n  Training {len(model_zoo)} models...')
    results = []
    for name, mdl in model_zoo.items():
        res = _evaluate(mdl, X_tr_sc, X_te_sc, y_tr, y_te, name)
        results.append(res)
        print(f'    ✅ {name:<20} R2={res["Test R2"]}  CV={res["CV R2 Mean"]}±{res["CV R2 Std"]}')

    results_df  = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
    results_df  = results_df.sort_values('Test R2', ascending=False).reset_index(drop=True)
    best_name   = results_df.iloc[0]['Model']
    print(f'\n  🏆 Best base model: {best_name}  (R²={results_df.iloc[0]["Test R2"]})')

    # 5. Optuna tuning
    print(f'\n  🔧 Optuna tuning ({best_name}, 30 trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: _optuna_objective(trial, best_name, X_tr_sc, y_tr),
                   n_trials=30, show_progress_bar=True)
    print(f'  ✅ Best CV R²: {study.best_value:.4f}  |  Params: {study.best_params}')

    # 6. Final model
    best_p = study.best_params.copy()
    if 'XGBoost' in best_name:
        best_p.update({'random_state': config.RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0})
        final_model = xgb.XGBRegressor(**best_p)
    elif 'LightGBM' in best_name and lgb is not None:
        best_p.update({'random_state': config.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
        final_model = lgb.LGBMRegressor(**best_p)
    else:
        best_p.update({'random_state': config.RANDOM_STATE, 'n_jobs': -1})
        final_model = RandomForestRegressor(**best_p)

    final_model.fit(X_tr_sc, y_tr)
    y_pred = final_model.predict(X_te_sc)
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f'\n  Final — R²: {r2:.4f} | MAE: {mae:,.1f} Kg | RMSE: {rmse:,.1f} Kg')

    # 7. Save using ModelArtifacts utility
    artifacts = ModelArtifacts()
    summary = {
        'model': best_name,
        'r2': round(r2, 4),
        'mae': round(mae, 1),
        'rmse': round(rmse, 1)
    }
    
    # Save all artifacts
    artifacts.save_sklearn_model(
        model=final_model,
        scaler=scaler,
        metadata=summary,
        base_path=config.MODEL_DIR_YIELD,
        label_encoders=label_encoders,
        features=config.YIELD_FEATURES
    )
    
    # Also save to summary.json for backward compatibility
    with open(config.YIELD_SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'  💾 Yield model saved → {config.YIELD_MODEL_PATH}')
    print(f'  ✅ Done! ({best_name}  R²={r2:.4f})')


if __name__ == '__main__':
    from setup_kaggle import setup_kaggle
    setup_kaggle()
    train()
