# 🌾 Rice AI Suite — Final Year Project

An end-to-end AI system for paddy/rice crop management combining four ML modules into a single Gradio web application.

---

## 📋 Modules

| Tab | Task | Algorithm |
|-----|------|-----------|
| 📈 Crop Yield | Predict paddy yield (Kg) from field parameters | XGBoost / LightGBM + Optuna tuning |
| 🍄 Disease Classifier | Classify paddy leaf diseases from images | EfficientNet-B3 (2-stage fine-tune) |
| 💧 Irrigation Predictor | Predict whether irrigation is needed | Random Forest (GridSearch tuned) |
| 🌱 Fertilizer Recommender | Recommend fertilizer type | Random Forest / Gradient Boosting |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (strongly recommended for the Disease module):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure credentials

Open `config.py` and set your credentials, **or** export them as environment variables:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
export GEMINI_API_KEY="your_gemini_api_key"   # optional — used for agronomic tips
```

> **Kaggle credentials:** Go to [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token.
> The `kaggle.json` file contains your `username` and `key`.
>
> **Disease dataset:** You must accept the competition rules at
> https://www.kaggle.com/competitions/paddy-disease-classification/rules
> **before** the download will work.

### 3. Run

```bash
python main.py
```

That's it. On the **first run**, `main.py` will:
1. Download all datasets from Kaggle (only if not already on disk)
2. Train all 4 models and save them to `./saved_models/`
3. Launch the Gradio web app at `http://127.0.0.1:7860`

On **every subsequent run**, datasets and models are loaded from disk — no re-downloading, no re-training.

---

## 🗂️ Project Structure

```
rice_ai_suite/
│
├── main.py              ← ✅ Single entry point — run this
├── app.py               ← Gradio UI (loads saved models)
├── config.py            ← All paths, seeds, API keys
├── setup_kaggle.py      ← Kaggle auth helper
├── requirements.txt
│
├── train_yield.py       ← Module 1: Crop Yield training
├── train_disease.py     ← Module 2: Disease Classification training
├── train_irrigation.py  ← Module 3: Irrigation Prediction training
├── train_fertilizer.py  ← Module 4: Fertilizer Recommendation training
│
├── saved_models/        ← Auto-created after first run
│   ├── yield/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoders.pkl
│   │   ├── features.pkl
│   │   └── summary.json
│   ├── disease/
│   │   ├── best.pth
│   │   ├── label_encoder.pkl
│   │   └── config.json
│   ├── irrigation/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   └── fertilizer/
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       └── feature_names.pkl
│
├── crop_data/           ← Yield dataset (auto-downloaded)
├── paddy_data/          ← Disease dataset (auto-downloaded)
├── data/                ← Irrigation dataset (auto-downloaded)
└── rice_fertilizer_data/← Fertilizer dataset (auto-downloaded)
```

---

## 🔁 Re-training a Specific Module

To force re-training of one module, simply delete its saved model folder:

```bash
rm -rf saved_models/yield/          # retrain yield model
rm -rf saved_models/disease/        # retrain disease model
rm -rf saved_models/irrigation/     # retrain irrigation model
rm -rf saved_models/fertilizer/     # retrain fertilizer model
```

Then run `python main.py` again.

To re-download a dataset (e.g., if files are corrupted):

```bash
rm -rf crop_data/                   # re-download yield dataset
rm -rf paddy_data/                  # re-download disease dataset
rm -rf data/                        # re-download irrigation dataset
rm -rf rice_fertilizer_data/        # re-download fertilizer dataset
```

---

## 🖥️ Hardware Notes

| Module | CPU | GPU |
|--------|-----|-----|
| Yield, Irrigation, Fertilizer | Fast (~5–15 min each) | Not needed |
| Disease (EfficientNet-B3) | Very slow (~hours) | Strongly recommended |

The system auto-detects CUDA. If no GPU is available, the disease model will train on CPU (slower but works).

---

## 📊 Datasets

| Module | Source |
|--------|--------|
| Crop Yield | [Kaggle: stealthtechnologies/predict-crop-production](https://www.kaggle.com/datasets/stealthtechnologies/predict-crop-production) |
| Disease | [Kaggle Competition: paddy-disease-classification](https://www.kaggle.com/competitions/paddy-disease-classification) |
| Irrigation | [Kaggle: pusainstitute/cropirrigationscheduling](https://www.kaggle.com/datasets/pusainstitute/cropirrigationscheduling) |
| Fertilizer | [Kaggle: hamzmiscof/rice-fertilizer-recommendation-dataset](https://www.kaggle.com/datasets/hamzmiscof/rice-fertilizer-recommendation-dataset) |

---

## 🤖 AI Features

- **Gemini AI** (Google) provides agronomic recommendation tips after yield predictions.
  Set `GEMINI_API_KEY` in `config.py` or as an environment variable.
  If unavailable, the app works normally without it.

---

## 🏫 Academic Note

This project is a final year project demonstrating:
- Regression (yield prediction with Optuna hyperparameter tuning)
- Multi-class image classification (EfficientNet-B3 transfer learning)
- Binary classification (irrigation scheduling)
- Multi-class tabular classification (fertilizer recommendation)
- Production-ready model persistence (train once, predict forever)
- Interactive Gradio web application
