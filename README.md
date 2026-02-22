# Rice AI Suite

Rice AI Suite is a unified Gradio app for paddy crop workflows with four ML modules:
- Crop Yield Prediction
- Disease Classification
- Irrigation Prediction
- Fertilizer Recommendation

## Installation Guide

1. Clone and enter the project
```bash
git clone <your-repo-url>
cd rice-suite
```

2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
```
Fill in at least:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `GEMINI_API_KEY` (optional, for AI tips)

## Run

First run (downloads datasets if missing, trains missing models, then launches app):
```bash
python main.py
```

Subsequent app-only run:
```bash
python app.py
```

## Re-Train / Re-Download

Re-train a specific module by removing its model folder:
```bash
rm -rf models/yield/
rm -rf models/disease/
rm -rf models/irrigation/
rm -rf models/fertilizer/
```

Re-download a specific dataset by removing its dataset folder:
```bash
rm -rf datasets/yield/
rm -rf datasets/disease/
rm -rf datasets/irrigation/
rm -rf datasets/fertilizer/
```

## Project Structure

```text
rice-suite/
├── main.py
├── app.py
├── config.py
├── setup_kaggle.py
├── load_env.py
├── requirements.txt
├── train_yield.py
├── train_disease.py
├── train_irrigation.py
├── train_fertilizer.py
├── models/
│   ├── yield/
│   ├── disease/
│   ├── irrigation/
│   └── fertilizer/
├── datasets/
│   ├── yield/
│   ├── disease/
│   ├── irrigation/
│   └── fertilizer/
├── src/
│   ├── data/
│   ├── services/
│   └── utils/
└── modules/
    └── *.ipynb
```
