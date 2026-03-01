"""
Gemini AI Service — Google Generative AI integration for agricultural recommendations.
"""
import google.generativeai as genai
import config


class GeminiService:
    """Service for generating AI-powered agricultural recommendations using Google Gemini."""

    def __init__(self):
        """Initialize Gemini API with credentials from config (if available)."""
        self.available = False
        try:
            api_key = str(config.GEMINI_API_KEY)
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                self.available = True
        except Exception:
            self.model = None
            self.available = False

    def generate_recommendation(self, prompt: str) -> str:
        """
        Generate agricultural recommendation based on prompt.

        Args:
            prompt: Natural language prompt describing the agricultural context

        Returns:
            AI-generated recommendation text, or error message if API fails
        """
        if not self.available:
            return "[Gemini API not configured - set GEMINI_API_KEY in .env for AI tips]"

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini unavailable: {e}]"

    # ── 1. Yield ──────────────────────────────────────────────────────────────

    def generate_yield_tip(self, prediction: float, hectares: float, inputs: dict) -> str:
        """Generate specific tip for improving yield based on prediction and inputs."""
        prompt = (
            f"Paddy yield predicted: {prediction:,.0f} Kg over {hectares:.1f} ha "
            f"({prediction / max(float(hectares), 0.1):,.0f} Kg/ha). "
            f"Inputs — LP: {inputs['lp']}T, DAP: {inputs['dap']}Kg, "
            f"Urea: {inputs['urea']}Kg, Pest: {inputs['pest']}ml, "
            f"Seed: {inputs['seed']}Kg, Rain: {inputs['rain']}mm. "
            "In ≤60 words, give one specific actionable tip to improve yield."
        )
        return self.generate_recommendation(prompt)

    # ── 2. Disease ────────────────────────────────────────────────────────────

    def generate_disease_tip(self, disease_name: str, confidence: float) -> str:
        """
        Generate field recommendation for a detected paddy disease.

        Args:
            disease_name: Human-readable disease name (e.g. 'Bacterial Leaf Blight')
            confidence: Model confidence as a percentage (0–100)

        Returns:
            AI-generated field action recommendation (≤60 words)
        """
        prompt = (
            f"A CNN model detected paddy disease: '{disease_name}' "
            f"with {confidence:.1f}% confidence. "
            "You are a rice plant pathologist. "
            "In ≤60 words, give ONE immediate, practical field action the farmer "
            "should take right now. If the plant is healthy (Normal), give a brief "
            "preventive care tip instead. No bullet points."
        )
        return self.generate_recommendation(prompt)

    # ── 3. Irrigation ─────────────────────────────────────────────────────────

    def generate_irrigation_tip(
        self,
        irrigation_needed: bool,
        confidence: float,
        stage: str,
        crop_days: int,
        soil_moisture: int,
        temperature: float,
        humidity: float,
    ) -> str:
        """
        Generate a watering recommendation based on irrigation prediction outputs.

        Args:
            irrigation_needed: Whether the model recommends irrigating
            confidence: Model confidence as a percentage (0–100)
            stage: Growth stage label (e.g. 'Tillering/Vegetative (21–60d)')
            crop_days: Age of the crop in days
            soil_moisture: Soil moisture sensor reading (100–800)
            temperature: Air temperature in °C
            humidity: Relative humidity in %

        Returns:
            AI-generated watering recommendation (≤60 words)
        """
        decision = "IRRIGATE NOW" if irrigation_needed else "HOLD — no irrigation needed"
        prompt = (
            f"Rice crop irrigation decision: {decision} (confidence {confidence:.1f}%). "
            f"Growth stage: {stage} ({crop_days} days old). "
            f"Soil moisture: {soil_moisture}, Temp: {temperature}°C, Humidity: {humidity}%. "
            "You are a rice irrigation specialist. "
            "In ≤60 words, give ONE specific watering recommendation for these exact "
            "conditions. Mention timing or quantity if relevant. No bullet points."
        )
        return self.generate_recommendation(prompt)

    # ── 4. Fertilizer ─────────────────────────────────────────────────────────

    def generate_fertilizer_tip(
        self,
        recommendation: str,
        confidence: float,
        growth_phase: str,
        soil_moisture: float,
        soil_temp: float,
        ec: float,
        air_temp: float,
        air_humidity: float,
    ) -> str:
        """
        Generate an application tip for the recommended fertilizer type.

        Args:
            recommendation: Recommended fertilizer name
            confidence: Model confidence as a percentage (0–100)
            growth_phase: 'Vegetative' or 'Reproductive'
            soil_moisture: Soil moisture in %
            soil_temp: Soil temperature in °C
            ec: Electrical conductivity in µS/cm
            air_temp: Air temperature in °C
            air_humidity: Air humidity in %

        Returns:
            AI-generated fertilizer application tip (≤60 words)
        """
        prompt = (
            f"Recommended fertilizer for rice: {recommendation} "
            f"(confidence {confidence:.1f}%, {growth_phase} phase). "
            f"Soil moisture: {soil_moisture:.1f}%, soil temp: {soil_temp:.1f}°C, "
            f"EC: {ec:.0f} µS/cm, air temp: {air_temp:.1f}°C, humidity: {air_humidity:.1f}%. "
            "You are a rice soil scientist. "
            "In ≤60 words, give ONE practical tip on how to apply this fertilizer "
            "under these exact conditions — timing, method, or dosage caution. "
            "No bullet points."
        )
        return self.generate_recommendation(prompt)