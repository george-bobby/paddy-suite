"""
main.py — Rice AI Suite: Master Entry Point
============================================
This is the ONE file you run to get everything working.

1. On FIRST run:
   - Downloads all datasets (only if not already downloaded)
   - Trains all 4 models (Yield, Disease, Irrigation, Fertilizer)
   - Saves models to ./models/

2. On SUBSEQUENT runs:
   - Skips downloading (datasets already on disk)
   - Skips training (models already saved)
   - Immediately launches the Gradio web app

To RETRAIN a specific model, delete its folder:
   - Yield      → delete ./models/yield/
   - Disease    → delete ./models/disease/
   - Irrigation → delete ./models/irrigation/
   - Fertilizer → delete ./models/fertilizer/

Usage:
    python main.py
    
Environment Setup:
    Requires KAGGLE_USERNAME, KAGGLE_KEY, and GEMINI_API_KEY environment variables.
    Create a .env file from .env.example or export them manually.
"""


# Load environment variables from .env file if present
try:
    from load_env import load_env_file
    load_env_file()
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")
    print("   Continuing with system environment variables...")


def main():
    print('='*65)
    print('  🌾  Rice AI Suite — Final Year Project')
    print('='*65)

    # 1. Kaggle setup (needed only for training / dataset downloads)
    print('\n📡 Setting up Kaggle API...')
    from setup_kaggle import setup_kaggle
    setup_kaggle()

    # 2. Train modules (each skips gracefully if already done)
    print('\n🧠 Checking / Training Models...')

    from train_yield       import train as train_yield
    from train_disease     import train as train_disease
    from train_irrigation  import train as train_irrigation
    from train_fertilizer  import train as train_fertilizer

    train_yield()
    train_disease()
    train_irrigation()
    train_fertilizer()

    print('\n' + '='*65)
    print('  ✅  All models ready — launching Gradio app...')
    print('='*65 + '\n')

    # 3. Launch app
    from app import build_app
    rice_app = build_app()
    rice_app.launch(share=False, debug=False)


if __name__ == '__main__':
    main()
