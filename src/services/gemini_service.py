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
    
    def generate_yield_tip(self, prediction: float, hectares: float, inputs: dict) -> str:
        """Generate specific tip for improving yield based on prediction and inputs."""
        prompt = (
            f"Paddy yield predicted: {prediction:,.0f} Kg over {hectares:.1f} ha "
            f"({prediction/max(float(hectares),0.1):,.0f} Kg/ha). "
            f"Inputs — LP: {inputs['lp']}T, DAP: {inputs['dap']}Kg, "
            f"Urea: {inputs['urea']}Kg, Pest: {inputs['pest']}ml, "
            f"Seed: {inputs['seed']}Kg, Rain: {inputs['rain']}mm. "
            "In ≤60 words, give one specific actionable tip to improve yield."
        )
        return self.generate_recommendation(prompt)
