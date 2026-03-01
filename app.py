"""
app.py — Rice AI Suite: 4-tab Gradio Interface
All four prediction functions now include Gemini AI recommendations.
Gemini section is hidden gracefully if API key is not configured.

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

if gemini_service.available:
    print('✅ All models loaded! (Gemini AI enabled)\n')
else:
    print('✅ All models loaded! (Gemini AI disabled — set GEMINI_API_KEY to enable)\n')

# Load yield dataset for slider bounds
_yield_df = pd.read_csv(config.YIELD_CSV)
_yield_df.columns = _yield_df.columns.str.strip()


# ─────────────────────── Helper ───────────────────────────────────────────────

def _ai_block(tip: str) -> str:
    """Wrap an AI tip in a styled markdown block. Returns empty string if Gemini is off."""
    if not gemini_service.available:
        return ''
    return f'\n\n---\n### 🤖 AI Recommendation\n\n{tip}'


# ─────────────────────── Prediction Functions ─────────────────────────────────

def predict_yield(lp, dap, urea, pest, seed, ha, rain):
    """Predict crop yield with optional Gemini AI recommendation."""
    try:
        result = yield_service.predict(lp, dap, urea, pest, seed, ha, rain)
        tip = yield_service.get_tip(result['prediction'], result['hectares'])

        ai_section = ''
        if gemini_service.available:
            ai_tip = gemini_service.generate_yield_tip(
                prediction=result['prediction'],
                hectares=result['hectares'],
                inputs={'lp': lp, 'dap': dap, 'urea': urea, 'pest': pest, 'seed': seed, 'rain': rain}
            )
            ai_section = _ai_block(ai_tip)

        return (
            f'## 🌾 Predicted Paddy Yield\n\n'
            f'| Metric | Value |\n|--------|-------|\n'
            f'| **Total Yield** | **{result["prediction"]:,.0f} Kg** |\n'
            f'| In Tonnes | {result["tonnes"]:.2f} T |\n'
            f'| Per Hectare | {result["per_hectare"]:,.0f} Kg/ha |\n'
            f'| 50-Kg Bags | ~{result["bags"]} bags |\n\n'
            f'{tip}'
            f'{ai_section}\n\n'
            f'> *Model: {result["summary"]["model"]}  |  '
            f'Test R² = {result["summary"]["r2"]:.4f}  |  '
            f'MAE = {result["summary"]["mae"]:,.0f} Kg*'
        )
    except Exception as e:
        return f'❌ Error: {e}'


def predict_disease(img):
    """Classify paddy disease with optional Gemini AI field recommendation."""
    if img is None:
        return "⚠️ Please upload an image to begin analysis."

    disease_info = {
        "bacterial_leaf_blight":    {"description": "A bacterial disease causing lesions on leaves, leading to drying and reduced yield.",                               "solution": "Use disease-free seeds, rotate crops, avoid excessive nitrogen, and apply recommended bactericides."},
        "bacterial_leaf_streak":    {"description": "Bacterial infection causing yellow streaks along leaf veins.",                                                      "solution": "Remove infected plants, use resistant varieties, and ensure proper spacing and hygiene."},
        "bacterial_panicle_blight": {"description": "Affects panicles, causing blight and poor grain formation.",                                                        "solution": "Plant resistant varieties, avoid high nitrogen fertilizers, and manage water effectively."},
        "blast":                    {"description": "Fungal disease causing diamond-shaped lesions, can destroy leaves, nodes, and panicles.",                           "solution": "Use resistant varieties, proper water management, and apply fungicides when necessary."},
        "brown_spot":               {"description": "Fungal infection causing brown spots on leaves and reduced grain quality.",                                          "solution": "Apply balanced fertilization, remove crop residues, and use fungicides if needed."},
        "dead_heart":               {"description": "Stem borer damage causing central leaf whorl to die.",                                                              "solution": "Use insect-resistant varieties, practice field sanitation, and apply recommended insecticides."},
        "downy_mildew":             {"description": "Fungal-like infection causing white downy growth on the underside of leaves.",                                       "solution": "Use resistant varieties, avoid waterlogging, and apply systemic fungicides."},
        "hispa":                    {"description": "Insect pest that scrapes leaf tissue causing silvery streaks.",                                                      "solution": "Use resistant varieties, maintain proper spacing, and apply insecticides if infestation is high."},
        "normal":                   {"description": "Healthy plant with no visible disease.",                                                                             "solution": "Maintain good crop management and monitor regularly."},
        "tungro":                   {"description": "Viral disease transmitted by leafhoppers, causing yellow-orange discoloration and stunted growth.",                  "solution": "Use resistant varieties, control leafhopper vectors, and remove infected plants."},
    }

    try:
        result      = disease_service.predict(img, top_k=1)
        top_class   = result["top_class"]
        emoji       = result.get("emoji", "🌾")
        confidence  = result["predictions"][0]["probability"] * 100

        info        = disease_info.get(top_class, {})
        description = info.get("description", "No description available.")
        solution    = info.get("solution", "No solution available.")

        ai_section = ''
        if gemini_service.available:
            ai_tip = gemini_service.generate_disease_tip(
                disease_name=top_class.replace('_', ' ').title(),
                confidence=confidence
            )
            ai_section = _ai_block(ai_tip)

        output  = f"## {emoji} {top_class.replace('_', ' ').title()}\n\n"
        output += f"**What it is:** {description}\n\n"
        output += f"**Standard Solution:** {solution}"
        output += ai_section
        return output

    except Exception as e:
        return f"❌ Error: {e}"


def predict_irrigation(crop_days, soil_moisture, temperature, humidity):
    """Predict irrigation need with optional Gemini AI recommendation."""
    try:
        result = irrigation_service.predict(crop_days, soil_moisture, temperature, humidity)

        decision       = '🌊 IRRIGATION NEEDED' if result['irrigation_needed'] else '✅ NO IRRIGATION NEEDED'
        confidence_str = f'Confidence: {result["confidence"]:.1f}%  |  Irrigation probability: {result["irrigation_probability"]:.1f}%'

        recommendations  = f'**Growth Stage:** {result["stage"]}\n\n'
        recommendations += '\n\n'.join(result['recommendations'])

        if gemini_service.available:
            ai_tip = gemini_service.generate_irrigation_tip(
                irrigation_needed=result['irrigation_needed'],
                confidence=result['confidence'],
                stage=result['stage'],
                crop_days=crop_days,
                soil_moisture=soil_moisture,
                temperature=temperature,
                humidity=humidity,
            )
            recommendations += _ai_block(ai_tip)

        return decision, confidence_str, recommendations

    except Exception as e:
        return f'❌ Error: {e}', '', ''


def predict_fertilizer(moist, soilT, EC, airT, airH, fase_tanam):
    """Recommend fertilizer type with optional Gemini AI application tip."""
    try:
        result      = fertilizer_service.predict(moist, soilT, EC, airT, airH, fase_tanam)
        icon        = fertilizer_service.get_confidence_icon(result['confidence'])
        phase_label = 'Vegetative' if fase_tanam == 0 else 'Reproductive'

        text = (
            f'### 🌾 Recommended: **{result["recommendation"]}**\n'
            f'### {icon} Confidence: **{result["confidence"]:.1f}%** ({result["confidence_level"]})\n\n'
            f'---\n\n'
        )

        if result['warning']:
            text += f'{result["warning"]}\n\n'

        if gemini_service.available:
            ai_tip = gemini_service.generate_fertilizer_tip(
                recommendation=result['recommendation'],
                confidence=result['confidence'],
                growth_phase=phase_label,
                soil_moisture=moist,
                soil_temp=soilT,
                ec=EC,
                air_temp=airT,
                air_humidity=airH,
            )
            text += _ai_block(ai_tip)

        return text

    except Exception as e:
        return f'❌ Error: {e}'


# ─────────────────────── Custom CSS ────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-900:  #0f2d14;
    --green-800:  #14401c;
    --green-700:  #1a5425;
    --green-600:  #216830;
    --green-500:  #2d8a42;
    --green-400:  #3db356;
    --green-200:  #a8e4b5;
    --green-50:   #f0fdf4;

    --surface-primary:   #ffffff;
    --surface-secondary: #f7fdf8;
    --surface-tertiary:  #edf7f0;
    --border-subtle:     #d1ead7;
    --border-strong:     #a3d4ac;

    --neutral-900: #171717;
    --neutral-800: #262626;
    --neutral-700: #404040;
    --neutral-500: #737373;

    --shadow-sm: 0 1px 3px rgba(21,84,37,0.08), 0 1px 2px rgba(21,84,37,0.04);
    --shadow-xl: 0 24px 56px rgba(21,84,37,0.18), 0 8px 24px rgba(21,84,37,0.10);

    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --radius-xl: 24px;

    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;
}

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: var(--font-body) !important;
    background: var(--surface-secondary) !important;
    color: var(--neutral-900) !important;
}

.rice-hero {
    background: linear-gradient(135deg, var(--green-800) 0%, var(--green-600) 50%, var(--green-500) 100%);
    border-radius: var(--radius-xl);
    padding: 48px 40px 40px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-xl);
}
.rice-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 70% 120%, rgba(201,162,39,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 80% at -10% 50%, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.rice-hero::after {
    content: '🌾';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 96px;
    opacity: 0.18;
    filter: blur(1px);
    pointer-events: none;
}
.rice-hero h1 {
    font-family: var(--font-display) !important;
    font-size: 2.6rem !important;
    font-weight: 400 !important;
    color: #ffffff !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px !important;
    line-height: 1.1 !important;
    text-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.rice-hero p {
    font-size: 1.05rem !important;
    color: var(--green-200) !important;
    margin: 0 !important;
    font-weight: 300 !important;
}

.tabs > .tab-nav {
    border-bottom: 2px solid var(--border-subtle) !important;
    background: transparent !important;
    gap: 4px !important;
    padding: 0 4px !important;
}
.tabs > .tab-nav button {
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--neutral-500) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    padding: 12px 20px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.tabs > .tab-nav button:hover {
    color: var(--green-600) !important;
    background: var(--green-50) !important;
}
.tabs > .tab-nav button.selected {
    color: var(--green-700) !important;
    border-bottom-color: var(--green-500) !important;
    background: var(--green-50) !important;
    font-weight: 600 !important;
}
.tabitem { padding: 28px 0 !important; }

.tab-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}
.tab-icon {
    width: 44px;
    height: 44px;
    border-radius: var(--radius-md);
    background: linear-gradient(135deg, var(--green-600), var(--green-400));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: var(--shadow-sm);
    flex-shrink: 0;
}
.tab-title-group h2 {
    font-family: var(--font-display) !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
    color: var(--green-900) !important;
    margin: 0 0 2px !important;
    line-height: 1.2 !important;
}
.tab-title-group p {
    font-size: 0.82rem !important;
    color: var(--neutral-500) !important;
    margin: 0 !important;
}

.card-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--green-600);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, var(--green-700) 0%, var(--green-500) 100%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: #fff !important;
    font-family: var(--font-body) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(33,104,48,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, var(--green-800) 0%, var(--green-600) 100%) !important;
    box-shadow: 0 6px 20px rgba(33,104,48,0.45) !important;
    transform: translateY(-1px) !important;
}
.gr-button-primary:active, button.primary:active {
    transform: translateY(0) !important;
}

input[type=range] { accent-color: var(--green-500) !important; }

.gr-textbox textarea, .gr-textbox input {
    border: 1.5px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    background: var(--surface-secondary) !important;
    color: var(--neutral-900) !important;
    transition: border-color 0.2s !important;
    padding: 10px 14px !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: var(--green-500) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(45,138,66,0.12) !important;
}

.gr-markdown { font-family: var(--font-body) !important; line-height: 1.65 !important; }
.gr-markdown h2 {
    font-family: var(--font-display) !important;
    color: var(--green-800) !important;
    font-size: 1.3rem !important;
    font-weight: 400 !important;
    margin-top: 0 !important;
}
.gr-markdown h3 {
    font-family: var(--font-display) !important;
    color: var(--green-700) !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
}
.gr-markdown table {
    border-collapse: collapse !important;
    width: 100% !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    margin: 12px 0 !important;
    box-shadow: var(--shadow-sm) !important;
}
.gr-markdown th {
    background: var(--green-700) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 14px !important;
    text-align: left !important;
}
.gr-markdown td {
    padding: 9px 14px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    font-size: 0.9rem !important;
}
.gr-markdown tr:nth-child(even) td { background: var(--green-50) !important; }
.gr-markdown tr:last-child td { border-bottom: none !important; }
.gr-markdown blockquote {
    border-left: 3px solid var(--green-400) !important;
    background: var(--green-50) !important;
    padding: 10px 16px !important;
    margin: 12px 0 !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
    color: var(--neutral-700) !important;
    font-size: 0.88rem !important;
}
.gr-markdown hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 16px 0 !important;
}

.gr-image {
    border: 2px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface-secondary) !important;
    transition: border-color 0.2s !important;
}
.gr-image:hover {
    border-color: var(--green-500) !important;
    background: var(--green-50) !important;
}

.gr-radio fieldset {
    border: 1.5px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 16px !important;
    background: var(--surface-secondary) !important;
}

.gr-label {
    border: 1.5px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface-primary) !important;
    overflow: hidden !important;
}

.output-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 48px 24px;
    text-align: center;
    color: var(--neutral-500);
}
.output-placeholder .icon { font-size: 2.5rem; opacity: 0.4; }
.output-placeholder p { font-size: 0.88rem; margin: 0; }

.rice-footer {
    text-align: center;
    padding: 20px 0 8px;
    color: var(--neutral-500);
    font-size: 0.78rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: 8px;
}
.rice-footer span {
    display: inline-block;
    margin: 0 6px;
    padding: 2px 8px;
    background: var(--surface-tertiary);
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--green-700);
}

@media (max-width: 768px) {
    .rice-hero { padding: 28px 20px 24px; }
    .rice-hero h1 { font-size: 1.9rem !important; }
    .rice-hero::after { display: none; }
}
"""


# ─────────────────────── Gradio UI ────────────────────────────────────────────

def build_app():
    # Dynamic labels based on Gemini availability
    gemini_badge  = ' ' if gemini_service.available else ''
    gemini_footer = '<span>Gemini 2.5 Flash</span>' if gemini_service.available else ''

    with gr.Blocks(
        title='🌾 Precision Farming For Rice',
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("DM Sans")],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
        ),
        css=CUSTOM_CSS
    ) as app:

        gr.HTML(f"""
        <div class="rice-hero">
            <h1>Precision Farming For Rice</h1>
            <p>Four AI-powered tools for intelligent crop management{gemini_badge}</p>
        </div>
        """)

        with gr.Tabs():

            # ─── TAB 1: YIELD ──────────────────────────────────────────────
            with gr.Tab('📈  Crop Yield'):
                gr.HTML(f"""
                <div class="tab-header">
                    <div class="tab-icon">📈</div>
                    <div class="tab-title-group">
                        <h2>Crop Yield Predictor</h2>
                        <p>Model: {yield_service.summary['model']} &nbsp;·&nbsp; Test R²: {yield_service.summary['r2']:.4f}{gemini_badge}</p>
                    </div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        gr.HTML('<div class="card-label">⚙️ Input Parameters</div>')
                        y_ha   = gr.Slider(float(_yield_df['Hectares'].min()), float(_yield_df['Hectares'].max()), value=float(_yield_df['Hectares'].median()), step=0.5, label="🗺️ Field Area (Hectares)")
                        y_rain = gr.Slider(float(_yield_df['30DRain( in mm)'].min()), float(_yield_df['30DRain( in mm)'].max()), value=float(_yield_df['30DRain( in mm)'].median()), step=0.5, label="🌧️ Rainfall — First 30 Days (mm)")
                        y_seed = gr.Slider(float(_yield_df['Seedrate(in Kg)'].min()), float(_yield_df['Seedrate(in Kg)'].max()), value=float(_yield_df['Seedrate(in Kg)'].median()), step=1.0, label="🫘 Seed Quantity (Kg)")
                        y_lp   = gr.Slider(float(_yield_df['LP_nurseryarea(in Tonnes)'].min()), float(_yield_df['LP_nurseryarea(in Tonnes)'].max()), value=float(_yield_df['LP_nurseryarea(in Tonnes)'].median()), step=0.5, label="🌱 Fertilizer in Nursery Area (Tonnes)")
                        y_dap  = gr.Slider(float(_yield_df['DAP_20days'].min()), float(_yield_df['DAP_20days'].max()), value=float(_yield_df['DAP_20days'].median()), step=1.0, label="🧪 DAP at 20 Days (Kg)")
                        y_pest = gr.Slider(float(_yield_df['Pest_60Day(in ml)'].min()), float(_yield_df['Pest_60Day(in ml)'].max()), value=float(_yield_df['Pest_60Day(in ml)'].median()), step=10.0, label="🐛 Pesticide at 60 Days (ml)")
                        y_urea = gr.Slider(float(_yield_df['Urea_40Days'].min()), float(_yield_df['Urea_40Days'].max()), value=float(_yield_df['Urea_40Days'].median()), step=1.0, label="🌿 Urea at 40 Days (Kg)")
                        y_btn  = gr.Button('🚀  Predict Yield', variant='primary')
                    with gr.Column(scale=2):
                        gr.HTML('<div class="card-label">📊 Prediction</div>')
                        y_out = gr.Markdown(
                            '<div class="output-placeholder"><div class="icon">🌾</div>'
                            '<p>Adjust parameters and click Predict.</p></div>'
                        )
                y_btn.click(fn=predict_yield, inputs=[y_lp, y_dap, y_urea, y_pest, y_seed, y_ha, y_rain], outputs=y_out)

            # ─── TAB 2: DISEASE ────────────────────────────────────────────
            with gr.Tab('🍄  Disease Classifier'):
                gr.HTML(f"""
                <div class="tab-header">
                    <div class="tab-icon">🍄</div>
                    <div class="tab-title-group">
                        <h2>Disease Detector</h2>
                        <p>EfficientNet-B3 · 10 disease categories{gemini_badge}</p>
                    </div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">📷 Upload Image</div>')
                        d_img = gr.Image(type='pil', label="Photo")
                        d_btn = gr.Button('🔍  Classify', variant='primary')
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">🔬 Diagnosis</div>')
                        d_out = gr.Markdown(
                            '<div class="output-placeholder"><div class="icon">🍃</div>'
                            '<p>Upload an image to receive a diagnosis.</p></div>'
                        )
                d_img.change(fn=predict_disease, inputs=d_img, outputs=d_out)
                d_btn.click(fn=predict_disease, inputs=d_img, outputs=d_out)

            # ─── TAB 3: IRRIGATION ─────────────────────────────────────────
            with gr.Tab('💧  Irrigation Advisor'):
                gr.HTML(f"""
                <div class="tab-header">
                    <div class="tab-icon">💧</div>
                    <div class="tab-title-group">
                        <h2>Irrigation Advisor</h2>
                        <p>Model: {irrigation_service.metadata['model_name']} &nbsp;·&nbsp; Accuracy: {irrigation_service.metadata['test_accuracy']:.2%}{gemini_badge}</p>
                    </div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">🌡️ Field Conditions</div>')
                        i_days = gr.Slider(1, 120, value=30, step=1, label="🌱 Crop Age (Days)")
                        i_mois = gr.Slider(100, 800, value=300, step=10, label="💧 Soil Moisture Level")
                        i_temp = gr.Slider(20, 42, value=30, step=1, label="🌡️ Temperature (°C)")
                        i_hum  = gr.Slider(10, 70, value=30, step=1, label="💨 Relative Humidity (%)")
                        i_btn  = gr.Button('🔮  Predict', variant='primary')
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">💡 Decision</div>')
                        i_result = gr.Textbox(label='Irrigation Decision', interactive=False)
                        i_conf   = gr.Textbox(label='Model Confidence', interactive=False)
                        i_rec    = gr.Markdown()
                i_btn.click(fn=predict_irrigation, inputs=[i_days, i_mois, i_temp, i_hum], outputs=[i_result, i_conf, i_rec])

            # ─── TAB 4: FERTILIZER ─────────────────────────────────────────
            with gr.Tab('🌱  Fertilizer Recommender'):
                gr.HTML(f"""
                <div class="tab-header">
                    <div class="tab-icon">🌱</div>
                    <div class="tab-title-group">
                        <h2>Fertilizer Recommender</h2>
                        <p>Soil &amp; environment analysis{gemini_badge}</p>
                    </div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">🧪 Soil &amp; Environment</div>')
                        f_moist = gr.Slider(20, 100, value=60, step=0.1, label="🌊 Soil Moisture (%)")
                        f_soilT = gr.Slider(20, 40, value=28, step=0.1, label="🌡️ Soil Temperature (°C)")
                        f_EC    = gr.Slider(0, 3500, value=1000, step=10, label="⚡ Electrical Conductivity (µS/cm)")
                        f_airT  = gr.Slider(20, 40, value=30, step=0.1, label="🌤️ Air Temperature (°C)")
                        f_airH  = gr.Slider(40, 100, value=75, step=0.1, label="💧 Air Humidity (%)")
                        f_fase  = gr.Radio([0, 1000], value=0, label="🌱 Growth Phase", info="0 = Vegetative Stage  |  1000 = Reproductive Stage")
                        f_btn   = gr.Button('🔮  Recommend', variant='primary')
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-label">📋 Recommendation</div>')
                        f_text = gr.Markdown(
                            '<div class="output-placeholder"><div class="icon">🌿</div>'
                            '<p>Enter soil conditions and click Recommend.</p></div>'
                        )
                f_btn.click(fn=predict_fertilizer, inputs=[f_moist, f_soilT, f_EC, f_airT, f_airH, f_fase], outputs=f_text)

        gr.HTML(f"""
        <div class="rice-footer">
            Precision Farming For Rice &nbsp;·&nbsp;
            <span>Scikit-learn</span><span>XGBoost</span><span>LightGBM</span>
            <span>EfficientNet-B3</span>{gemini_footer}<span>Gradio</span>
        </div>
        """)

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(share=False, debug=False)