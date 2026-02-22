"""
app.py — Rice AI Suite: 4-tab Gradio Interface
Loads all 4 trained models and launches the Gradio web app.
Models are loaded once at startup; predictions are fast on subsequent runs.

Usage:
    python app.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torchvision.transforms as T
from torch.cuda.amp import autocast
import timm
import gradio as gr
import google.generativeai as genai

import config

warnings.filterwarnings('ignore')

# ─────────────────────── Gemini ───────────────────────────────────────────────

genai.configure(api_key=config.GEMINI_API_KEY)
_gemini = genai.GenerativeModel("gemini-2.5-flash")

def _gemini_call(prompt: str) -> str:
    try:
        return _gemini.generate_content(prompt).text.strip()
    except Exception as e:
        return f"[Gemini unavailable: {e}]"


# ─────────────────────── Load yield model ────────────────────────────────────

def _load_yield():
    model    = joblib.load(config.YIELD_MODEL_PATH)
    scaler   = joblib.load(config.YIELD_SCALER_PATH)
    features = joblib.load(config.YIELD_FEATURES_PATH)
    with open(config.YIELD_SUMMARY_PATH) as f:
        summary = json.load(f)
    return model, scaler, features, summary


# ─────────────────────── Load disease model ──────────────────────────────────

def _load_disease():
    with open(config.DISEASE_CONFIG_PATH) as f:
        cfg = json.load(f)
    with open(config.DISEASE_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=cfg['num_classes'])
    model.load_state_dict(
        torch.load(config.DISEASE_MODEL_PATH, map_location=config.DEVICE)
    )
    model = model.to(config.DEVICE)
    model.eval()
    return model, le, cfg


# ─────────────────────── Load irrigation model ───────────────────────────────

def _load_irrigation():
    model  = joblib.load(config.IRR_MODEL_PATH)
    scaler = joblib.load(config.IRR_SCALER_PATH)
    with open(config.IRR_METADATA_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta


# ─────────────────────── Load fertilizer model ───────────────────────────────

def _load_fertilizer():
    model    = joblib.load(config.FERT_MODEL_PATH)
    scaler   = joblib.load(config.FERT_SCALER_PATH)
    le       = joblib.load(config.FERT_ENCODER_PATH)
    features = joblib.load(config.FERT_FEATURES_PATH)
    return model, scaler, le, features


# ─────────────────────── Startup: load all models ────────────────────────────

print('\n🔄 Loading all trained models...')
YIELD_MODEL, YIELD_SCALER, YIELD_FEATURES, YIELD_SUMMARY = _load_yield()
DISEASE_MODEL, DISEASE_LE, DISEASE_CFG = _load_disease()
IRR_MODEL, IRR_SCALER, IRR_META = _load_irrigation()
FERT_MODEL, FERT_SCALER, FERT_LE, FERT_FEATURES = _load_fertilizer()
print('✅ All models loaded!\n')

# Load yield dataset for slider bounds
_yield_df = pd.read_csv(config.YIELD_CSV)
_yield_df.columns = _yield_df.columns.str.strip()

DISEASE_INFO = {
    'bacterial_leaf_blight'   : ('💧', 'Water-soaked to yellowish stripe on leaf margins.'),
    'bacterial_leaf_streak'   : ('💧', 'Dark brown streaks with wavy margins on leaves.'),
    'bacterial_panicle_blight': ('💧', 'Grain discoloration & panicle sterility.'),
    'blast'                   : ('🍄', 'Diamond-shaped lesions with gray centers.'),
    'brown_spot'              : ('🟤', 'Circular brown spots with yellow halo.'),
    'dead_heart'              : ('☠️', 'Central shoot dies — caused by stem borers.'),
    'downy_mildew'            : ('🌫️', 'Yellowish patches, white fungal growth below.'),
    'hispa'                   : ('🐛', 'White blotches caused by leaf mining larvae.'),
    'normal'                  : ('✅', 'Healthy paddy plant — no disease detected.'),
    'tungro'                  : ('🟡', 'Yellow-orange discoloration, stunted growth.'),
}


# ─────────────────────── Prediction functions ─────────────────────────────────

def _yield_eng(row: dict) -> dict:
    h  = row.get('Hectares', 1)
    lp = row.get('LP_nurseryarea(in Tonnes)', 0)
    row['Fertilizer_per_Ha'] = lp / (h + 1e-5)
    row['Input_Intensity']   = row.get('DAP_20days', 0) + row.get('Pest_60Day(in ml)', 0) / 10 + lp * 100
    row['Seed_Density']      = row.get('Seedrate(in Kg)', 0) / (h + 1e-5)
    row['Rain_per_Ha']       = row.get('30DRain( in mm)', 0) / (h + 1e-5)
    return row


def predict_yield(lp, dap, urea, pest, seed, ha, rain):
    try:
        row  = {'LP_nurseryarea(in Tonnes)': float(lp), 'DAP_20days': float(dap),
                'Urea_40Days': float(urea), 'Pest_60Day(in ml)': float(pest),
                'Seedrate(in Kg)': float(seed), 'Hectares': float(ha),
                '30DRain( in mm)': float(rain)}
        row  = _yield_eng(row)
        X_in = pd.DataFrame([row])[YIELD_FEATURES]
        pred = float(YIELD_MODEL.predict(YIELD_SCALER.transform(X_in))[0])
        tip  = ('⚠️ Low yield. Adjust fertilizer/pesticide.' if pred < 20000 else
                '🌟 Excellent yield! Inputs well optimised.'  if pred > 42000 else
                '👍 Good yield. Small DAP/pest tweaks may help.')

        # Optional Gemini recommendation
        prompt = (f"Paddy yield predicted: {pred:,.0f} Kg over {ha:.1f} ha "
                  f"({pred/max(float(ha),0.1):,.0f} Kg/ha). "
                  f"Inputs — LP: {lp}T, DAP: {dap}Kg, Urea: {urea}Kg, Pest: {pest}ml, "
                  f"Seed: {seed}Kg, Rain: {rain}mm. "
                  "In ≤60 words, give one specific actionable tip to improve yield.")
        ai_tip = _gemini_call(prompt)

        return (f'## 🌾 Predicted Paddy Yield\n\n'
                f'| Metric | Value |\n|--------|-------|\n'
                f'| **Total Yield** | **{pred:,.0f} Kg** |\n'
                f'| In Tonnes | {pred/1000:.2f} T |\n'
                f'| Per Hectare | {pred/max(float(ha),0.1):,.0f} Kg/ha |\n'
                f'| 50-Kg Bags | ~{int(pred/50)} bags |\n\n'
                f'{tip}\n\n'
                f'---\n🤖 **AI Tip:** {ai_tip}\n\n'
                f'> *Model: {YIELD_SUMMARY["model"]}  |  '
                f'Test R² = {YIELD_SUMMARY["r2"]:.4f}  |  '
                f'MAE = {YIELD_SUMMARY["mae"]:,.0f} Kg*')
    except Exception as e:
        return f'❌ Error: {e}'


def predict_disease(img):
    if img is None:
        return '⚠️ Please upload an image.'
    tfm = T.Compose([
        T.Resize((DISEASE_CFG['img_size'], DISEASE_CFG['img_size'])),
        T.ToTensor(),
        T.Normalize(config.IMG_MEAN, config.IMG_STD),
    ])
    inp = tfm(img).unsqueeze(0).to(config.DEVICE)
    DISEASE_MODEL.eval()
    with torch.no_grad(), autocast():
        probs = torch.softmax(DISEASE_MODEL(inp), dim=1)[0]
    top3_probs, top3_idx = probs.topk(3)
    top_cls     = DISEASE_LE.classes_[top3_idx[0].item()]
    emoji, desc = DISEASE_INFO.get(top_cls, ('🌾', ''))
    lines = [f'## {emoji} {top_cls.replace("_", " ").title()}', desc, '', '---', '### Top 3 Predictions']
    for prob, idx in zip(top3_probs, top3_idx):
        cls = DISEASE_LE.classes_[idx.item()].replace('_', ' ').title()
        lines.append(f'**{cls}**: {prob.item()*100:.1f}% {"█" * int(prob.item()*30)}')
    return '\n'.join(lines)


def predict_irrigation(crop_days, soil_moisture, temperature, humidity):
    try:
        inp  = np.array([[crop_days, soil_moisture, temperature, humidity]])
        if IRR_META.get('needs_scale', False):
            inp = IRR_SCALER.transform(inp)
        pred  = IRR_MODEL.predict(inp)[0]
        proba = IRR_MODEL.predict_proba(inp)[0]
        conf  = proba[pred] * 100

        result = '🌊 IRRIGATION NEEDED' if pred == 1 else '✅ NO IRRIGATION NEEDED'
        conf_str = f'Confidence: {conf:.1f}%  |  Irrigation probability: {proba[1]*100:.1f}%'

        if crop_days <= 20:   stage, note = 'Transplanting/Establishment (1–20d)', '🌱 Critical stage — consistent water needed.'
        elif crop_days <= 60: stage, note = 'Tillering/Vegetative (21–60d)',       '🌿 Maintain moisture for tillering.'
        elif crop_days <= 90: stage, note = 'Panicle/Heading (61–90d)',            '🌾 Water stress now reduces yield significantly.'
        else:                 stage, note = 'Grain Fill/Maturation (90+d)',        '🍚 Water demand reduces at this stage.'

        recs = [note]
        if soil_moisture < 200:   recs.append(f'💧 Soil moisture {soil_moisture} critically low — irrigate immediately.')
        elif soil_moisture < 350: recs.append(f'💧 Soil moisture {soil_moisture} low — irrigation recommended.')
        elif soil_moisture < 500: recs.append(f'💧 Soil moisture {soil_moisture} moderate — monitor closely.')
        else:                     recs.append(f'💧 Soil moisture {soil_moisture} adequate.')
        if temperature > 36: recs.append(f'🌡️ Temp {temperature}°C extreme — high evapotranspiration.')
        elif temperature > 32: recs.append(f'🌡️ Temp {temperature}°C high — increased water demand.')
        if humidity < 20: recs.append(f'💨 Humidity {humidity}% very low — irrigate more frequently.')
        elif humidity > 60: recs.append(f'💨 Humidity {humidity}% high — less irrigation may suffice.')

        rec_md = f'**Growth Stage:** {stage}\n\n' + '\n\n'.join(recs)
        return result, conf_str, rec_md
    except Exception as e:
        return f'❌ Error: {e}', '', ''


def predict_fertilizer(moist, soilT, EC, airT, airH, fase_tanam):
    try:
        inp_sc   = FERT_SCALER.transform(np.array([[moist, soilT, EC, airT, airH, fase_tanam]]))
        pred_enc = FERT_MODEL.predict(inp_sc)[0]
        proba    = FERT_MODEL.predict_proba(inp_sc)[0]
        pred     = FERT_LE.inverse_transform([pred_enc])[0]
        conf     = proba[pred_enc] * 100

        probs_dict = {FERT_LE.classes_[i]: float(p * 100) for i, p in enumerate(proba)}
        sorted_p   = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))

        icon  = '🟢' if conf >= 70 else '🟡' if conf >= 50 else '🔴'
        level = 'High'  if conf >= 70 else 'Medium' if conf >= 50 else 'Low'
        txt   = f'### 🌾 Recommended: **{pred}**\n### {icon} Confidence: **{conf:.1f}%** ({level})\n\n---\n\n'
        txt  += '#### All Probabilities:\n\n'
        for fert, prob in sorted_p.items():
            txt += f'**{fert}**: {prob:.1f}% {"█" * int(prob / 2)}\n\n'
        if conf < 60:
            txt += '\n⚠️ Low confidence — consult an agronomist.'
        return txt, sorted_p
    except Exception as e:
        return f'❌ Error: {e}', {}


# ─────────────────────── Gradio UI ────────────────────────────────────────────

def build_app():
    with gr.Blocks(title='🌾 Rice AI Suite', theme=gr.themes.Soft()) as app:

        gr.Markdown("""
# 🌾 Rice AI Suite
### Four AI-powered tools for paddy crop management — all in one place
---
""")

        with gr.Tabs():

            # ─── TAB 1: YIELD ──────────────────────────────────────────────
            with gr.Tab('📈 Crop Yield'):
                gr.Markdown(f"""
### 🌾 Paddy Crop Yield Predictor
**Model:** {YIELD_SUMMARY['model']}  |  **Test R²:** {YIELD_SUMMARY['r2']:.4f}
""")
                with gr.Row():
                    with gr.Column(scale=3):
                        y_ha   = gr.Slider(float(_yield_df['Hectares'].min()),
                                           float(_yield_df['Hectares'].max()),
                                           value=float(_yield_df['Hectares'].median()), step=0.5,
                                           label="🗺️ Field Area (Hectares)")
                        y_rain = gr.Slider(float(_yield_df['30DRain( in mm)'].min()),
                                           float(_yield_df['30DRain( in mm)'].max()),
                                           value=float(_yield_df['30DRain( in mm)'].median()), step=0.5,
                                           label="🌧️ Rainfall — First 30 Days (mm)")
                        y_seed = gr.Slider(float(_yield_df['Seedrate(in Kg)'].min()),
                                           float(_yield_df['Seedrate(in Kg)'].max()),
                                           value=float(_yield_df['Seedrate(in Kg)'].median()), step=1.0,
                                           label="🫘 Seed Quantity (Kg)")
                        y_lp   = gr.Slider(float(_yield_df['LP_nurseryarea(in Tonnes)'].min()),
                                           float(_yield_df['LP_nurseryarea(in Tonnes)'].max()),
                                           value=float(_yield_df['LP_nurseryarea(in Tonnes)'].median()), step=0.5,
                                           label="🌱 Fertilizer in Nursery Area (Tonnes)")
                        y_dap  = gr.Slider(float(_yield_df['DAP_20days'].min()),
                                           float(_yield_df['DAP_20days'].max()),
                                           value=float(_yield_df['DAP_20days'].median()), step=1.0,
                                           label="🧪 DAP at 20 Days (Kg)")
                        y_pest = gr.Slider(float(_yield_df['Pest_60Day(in ml)'].min()),
                                           float(_yield_df['Pest_60Day(in ml)'].max()),
                                           value=float(_yield_df['Pest_60Day(in ml)'].median()), step=10.0,
                                           label="🐛 Pesticide at 60 Days (ml)")
                        y_urea = gr.Slider(float(_yield_df['Urea_40Days'].min()),
                                           float(_yield_df['Urea_40Days'].max()),
                                           value=float(_yield_df['Urea_40Days'].median()), step=1.0,
                                           label="🌿 Urea at 40 Days (Kg)")
                        y_btn  = gr.Button('🚀 Predict Yield', variant='primary')
                    with gr.Column(scale=2):
                        y_out  = gr.Markdown('*Adjust sliders and click Predict.*')

                y_btn.click(fn=predict_yield,
                            inputs=[y_lp, y_dap, y_urea, y_pest, y_seed, y_ha, y_rain],
                            outputs=y_out)

            # ─── TAB 2: DISEASE ────────────────────────────────────────────
            with gr.Tab('🍄 Disease Classifier'):
                gr.Markdown("### 🍄 Paddy Disease Classifier")
                with gr.Row():
                    with gr.Column():
                        d_img = gr.Image(type='pil', label="📷 Upload Paddy Leaf Image")
                        d_btn = gr.Button('🔍 Classify Disease', variant='primary')
                    with gr.Column():
                        d_out = gr.Markdown('*Upload an image to classify.*')
                d_img.change(fn=predict_disease, inputs=d_img, outputs=d_out)
                d_btn.click(fn=predict_disease, inputs=d_img, outputs=d_out)

            # ─── TAB 3: IRRIGATION ─────────────────────────────────────────
            with gr.Tab('💧 Irrigation Predictor'):
                gr.Markdown(f"""
### 💧 Paddy Irrigation Prediction
**Model:** {IRR_META['model_name']}  |  **Accuracy:** {IRR_META['test_accuracy']:.2%}
""")
                with gr.Row():
                    with gr.Column():
                        i_days = gr.Slider(1, 120, value=30, step=1,  label="🌱 Crop Days")
                        i_mois = gr.Slider(100, 800, value=300, step=10, label="💧 Soil Moisture")
                        i_temp = gr.Slider(20, 42, value=30, step=1,  label="🌡️ Temperature (°C)")
                        i_hum  = gr.Slider(10, 70, value=30, step=1,  label="💨 Humidity (%)")
                        i_btn  = gr.Button('🔮 Predict Irrigation', variant='primary')
                    with gr.Column():
                        i_result = gr.Textbox(label='Irrigation Decision', interactive=False)
                        i_conf   = gr.Textbox(label='Probability', interactive=False)
                        i_rec    = gr.Markdown()
                i_btn.click(fn=predict_irrigation,
                            inputs=[i_days, i_mois, i_temp, i_hum],
                            outputs=[i_result, i_conf, i_rec])

            # ─── TAB 4: FERTILIZER ─────────────────────────────────────────
            with gr.Tab('🌱 Fertilizer Recommender'):
                gr.Markdown("### 🌱 Rice Fertilizer Recommendation")
                with gr.Row():
                    with gr.Column():
                        f_moist = gr.Slider(20, 100, value=60, step=0.1,  label="🌊 Soil Moisture (%)")
                        f_soilT = gr.Slider(20, 40,  value=28, step=0.1,  label="🌡️ Soil Temperature (°C)")
                        f_EC    = gr.Slider(0, 3500,  value=1000, step=10, label="⚡ Electrical Conductivity (µS/cm)")
                        f_airT  = gr.Slider(20, 40,  value=30, step=0.1,  label="🌡️ Air Temperature (°C)")
                        f_airH  = gr.Slider(40, 100, value=75, step=0.1,  label="💧 Air Humidity (%)")
                        f_fase  = gr.Radio([0, 1000], value=0,
                                           label="🌱 Growth Phase (0=Vegetative | 1000=Reproductive)")
                        f_btn   = gr.Button('🔮 Get Recommendation', variant='primary')
                    with gr.Column():
                        f_text = gr.Markdown()
                        f_prob = gr.Label(num_top_classes=6)
                f_btn.click(fn=predict_fertilizer,
                            inputs=[f_moist, f_soilT, f_EC, f_airT, f_airH, f_fase],
                            outputs=[f_text, f_prob])

        gr.Markdown('---\n*Rice AI Suite — Scikit-learn · XGBoost · LightGBM · PyTorch · Gradio*')

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(share=False, debug=False)
