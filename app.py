"""
app.py — Rice AI Suite: 4-tab Gradio Interface
Refactored version using service classes for clean separation of concerns.

Usage:
    python app.py
"""

import warnings
import pandas as pd
import gradio as gr

import config
from src.services import (
    GeminiService,
    YieldService,
    DiseaseService,
    IrrigationService,
    FertilizerService
)

warnings.filterwarnings('ignore')

# ─────────────────────── Initialize Services ─────────────────────────────────

print('\n🔄 Loading all trained models...')
gemini_service = GeminiService()
yield_service = YieldService()
disease_service = DiseaseService()
irrigation_service = IrrigationService()
fertilizer_service = FertilizerService()
print('✅ All models loaded!\n')

# Load yield dataset for slider bounds
_yield_df = pd.read_csv(config.YIELD_CSV)
_yield_df.columns = _yield_df.columns.str.strip()


# ─────────────────────── Prediction Functions ─────────────────────────────────

def predict_yield(lp, dap, urea, pest, seed, ha, rain):
    """Predict crop yield with AI recommendations."""
    try:
        # Get prediction
        result = yield_service.predict(lp, dap, urea, pest, seed, ha, rain)
        
        # Get simple tip
        tip = yield_service.get_tip(result['prediction'], result['hectares'])
        
        # Get AI recommendation
        ai_tip = gemini_service.generate_yield_tip(
            prediction=result['prediction'],
            hectares=result['hectares'],
            inputs={'lp': lp, 'dap': dap, 'urea': urea, 'pest': pest, 'seed': seed, 'rain': rain}
        )
        
        # Format output
        return (
            f'## 🌾 Predicted Paddy Yield\n\n'
            f'| Metric | Value |\n|--------|-------|\n'
            f'| **Total Yield** | **{result["prediction"]:,.0f} Kg** |\n'
            f'| In Tonnes | {result["tonnes"]:.2f} T |\n'
            f'| Per Hectare | {result["per_hectare"]:,.0f} Kg/ha |\n'
            f'| 50-Kg Bags | ~{result["bags"]} bags |\n\n'
            f'{tip}\n\n'
            f'---\n🤖 **AI Tip:** {ai_tip}\n\n'
            f'> *Model: {result["summary"]["model"]}  |  '
            f'Test R² = {result["summary"]["r2"]:.4f}  |  '
            f'MAE = {result["summary"]["mae"]:,.0f} Kg*'
        )
    except Exception as e:
        return f'❌ Error: {e}'


def predict_disease(img):
    """Classify paddy disease from leaf image."""
    if img is None:
        return '⚠️ Please upload an image.'
    
    try:
        result = disease_service.predict(img, top_k=3)
        
        lines = [
            f'## {result["emoji"]} {result["top_class"].replace("_", " ").title()}',
            result['description'],
            '',
            '---',
            '### Top 3 Predictions'
        ]
        
        for pred in result['predictions']:
            prob = pred['probability'] * 100
            cls = pred['class'].replace('_', ' ').title()
            bar = "█" * int(prob / 3.33)  # Scale to ~30 chars max
            lines.append(f'**{cls}**: {prob:.1f}% {bar}')
        
        return '\n'.join(lines)
    except Exception as e:
        return f'❌ Error: {e}'


def predict_irrigation(crop_days, soil_moisture, temperature, humidity):
    """Predict irrigation requirements."""
    try:
        result = irrigation_service.predict(crop_days, soil_moisture, temperature, humidity)
        
        decision = '🌊 IRRIGATION NEEDED' if result['irrigation_needed'] else '✅ NO IRRIGATION NEEDED'
        confidence = f'Confidence: {result["confidence"]:.1f}%  |  Irrigation probability: {result["irrigation_probability"]:.1f}%'
        
        recommendations = f'**Growth Stage:** {result["stage"]}\n\n'
        recommendations += '\n\n'.join(result['recommendations'])
        
        return decision, confidence, recommendations
    except Exception as e:
        return f'❌ Error: {e}', '', ''


def predict_fertilizer(moist, soilT, EC, airT, airH, fase_tanam):
    """Recommend optimal fertilizer type."""
    try:
        result = fertilizer_service.predict(moist, soilT, EC, airT, airH, fase_tanam)
        
        icon = fertilizer_service.get_confidence_icon(result['confidence'])
        
        text = (
            f'### 🌾 Recommended: **{result["recommendation"]}**\n'
            f'### {icon} Confidence: **{result["confidence"]:.1f}%** ({result["confidence_level"]})\n\n'
            f'---\n\n'
            f'#### All Probabilities:\n\n'
        )
        
        for fert, prob in result['probabilities'].items():
            text += f'**{fert}**: {prob:.1f}% {"█" * int(prob / 2)}\n\n'
        
        if result['warning']:
            text += f'\n{result["warning"]}'
        
        return text, result['probabilities']
    except Exception as e:
        return f'❌ Error: {e}', {}


# ─────────────────────── Gradio UI ────────────────────────────────────────────

def build_app():
    """Build and return Gradio app interface."""
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
**Model:** {yield_service.summary['model']}  |  **Test R²:** {yield_service.summary['r2']:.4f}
""")
                with gr.Row():
                    with gr.Column(scale=3):
                        y_ha = gr.Slider(
                            float(_yield_df['Hectares'].min()),
                            float(_yield_df['Hectares'].max()),
                            value=float(_yield_df['Hectares'].median()),
                            step=0.5,
                            label="🗺️ Field Area (Hectares)"
                        )
                        y_rain = gr.Slider(
                            float(_yield_df['30DRain( in mm)'].min()),
                            float(_yield_df['30DRain( in mm)'].max()),
                            value=float(_yield_df['30DRain( in mm)'].median()),
                            step=0.5,
                            label="🌧️ Rainfall — First 30 Days (mm)"
                        )
                        y_seed = gr.Slider(
                            float(_yield_df['Seedrate(in Kg)'].min()),
                            float(_yield_df['Seedrate(in Kg)'].max()),
                            value=float(_yield_df['Seedrate(in Kg)'].median()),
                            step=1.0,
                            label="🫘 Seed Quantity (Kg)"
                        )
                        y_lp = gr.Slider(
                            float(_yield_df['LP_nurseryarea(in Tonnes)'].min()),
                            float(_yield_df['LP_nurseryarea(in Tonnes)'].max()),
                            value=float(_yield_df['LP_nurseryarea(in Tonnes)'].median()),
                            step=0.5,
                            label="🌱 Fertilizer in Nursery Area (Tonnes)"
                        )
                        y_dap = gr.Slider(
                            float(_yield_df['DAP_20days'].min()),
                            float(_yield_df['DAP_20days'].max()),
                            value=float(_yield_df['DAP_20days'].median()),
                            step=1.0,
                            label="🧪 DAP at 20 Days (Kg)"
                        )
                        y_pest = gr.Slider(
                            float(_yield_df['Pest_60Day(in ml)'].min()),
                            float(_yield_df['Pest_60Day(in ml)'].max()),
                            value=float(_yield_df['Pest_60Day(in ml)'].median()),
                            step=10.0,
                            label="🐛 Pesticide at 60 Days (ml)"
                        )
                        y_urea = gr.Slider(
                            float(_yield_df['Urea_40Days'].min()),
                            float(_yield_df['Urea_40Days'].max()),
                            value=float(_yield_df['Urea_40Days'].median()),
                            step=1.0,
                            label="🌿 Urea at 40 Days (Kg)"
                        )
                        y_btn = gr.Button('🚀 Predict Yield', variant='primary')
                    with gr.Column(scale=2):
                        y_out = gr.Markdown('*Adjust sliders and click Predict.*')

                y_btn.click(
                    fn=predict_yield,
                    inputs=[y_lp, y_dap, y_urea, y_pest, y_seed, y_ha, y_rain],
                    outputs=y_out
                )

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
**Model:** {irrigation_service.metadata['model_name']}  |  **Accuracy:** {irrigation_service.metadata['test_accuracy']:.2%}
""")
                with gr.Row():
                    with gr.Column():
                        i_days = gr.Slider(1, 120, value=30, step=1, label="🌱 Crop Days")
                        i_mois = gr.Slider(100, 800, value=300, step=10, label="💧 Soil Moisture")
                        i_temp = gr.Slider(20, 42, value=30, step=1, label="🌡️ Temperature (°C)")
                        i_hum = gr.Slider(10, 70, value=30, step=1, label="💨 Humidity (%)")
                        i_btn = gr.Button('🔮 Predict Irrigation', variant='primary')
                    with gr.Column():
                        i_result = gr.Textbox(label='Irrigation Decision', interactive=False)
                        i_conf = gr.Textbox(label='Probability', interactive=False)
                        i_rec = gr.Markdown()
                i_btn.click(
                    fn=predict_irrigation,
                    inputs=[i_days, i_mois, i_temp, i_hum],
                    outputs=[i_result, i_conf, i_rec]
                )

            # ─── TAB 4: FERTILIZER ─────────────────────────────────────────
            with gr.Tab('🌱 Fertilizer Recommender'):
                gr.Markdown("### 🌱 Rice Fertilizer Recommendation")
                with gr.Row():
                    with gr.Column():
                        f_moist = gr.Slider(20, 100, value=60, step=0.1, label="🌊 Soil Moisture (%)")
                        f_soilT = gr.Slider(20, 40, value=28, step=0.1, label="🌡️ Soil Temperature (°C)")
                        f_EC = gr.Slider(0, 3500, value=1000, step=10, label="⚡ Electrical Conductivity (µS/cm)")
                        f_airT = gr.Slider(20, 40, value=30, step=0.1, label="🌡️ Air Temperature (°C)")
                        f_airH = gr.Slider(40, 100, value=75, step=0.1, label="💧 Air Humidity (%)")
                        f_fase = gr.Radio(
                            [0, 1000],
                            value=0,
                            label="🌱 Growth Phase (0=Vegetative | 1000=Reproductive)"
                        )
                        f_btn = gr.Button('🔮 Get Recommendation', variant='primary')
                    with gr.Column():
                        f_text = gr.Markdown()
                        f_prob = gr.Label(num_top_classes=6)
                f_btn.click(
                    fn=predict_fertilizer,
                    inputs=[f_moist, f_soilT, f_EC, f_airT, f_airH, f_fase],
                    outputs=[f_text, f_prob]
                )

        gr.Markdown('---\n*Rice AI Suite v2.0 — Scikit-learn · XGBoost · LightGBM · PyTorch · Gradio*')

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(share=False, debug=False)
