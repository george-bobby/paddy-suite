"""
Yield Prediction Service — Paddy crop yield forecasting.
"""
import json
import joblib
import pandas as pd
import config


class YieldService:
    """Service for predicting paddy crop yield based on input parameters."""
    
    def __init__(self):
        """Load trained yield prediction model and artifacts."""
        self.model = joblib.load(config.YIELD_MODEL_PATH)
        self.scaler = joblib.load(config.YIELD_SCALER_PATH)
        self.features = joblib.load(config.YIELD_FEATURES_PATH)
        with open(config.YIELD_SUMMARY_PATH) as f:
            self.summary = json.load(f)
    
    @staticmethod
    def engineer_features(row: dict) -> dict:
        """
        Create engineered features from raw input.
        
        Args:
            row: Dictionary with input parameters
            
        Returns:
            Dictionary with added engineered features
        """
        h = row.get('Hectares', 1)
        lp = row.get('LP_nurseryarea(in Tonnes)', 0)
        row['Fertilizer_per_Ha'] = lp / (h + 1e-5)
        row['Input_Intensity'] = (
            row.get('DAP_20days', 0) + 
            row.get('Pest_60Day(in ml)', 0) / 10 + 
            lp * 100
        )
        row['Seed_Density'] = row.get('Seedrate(in Kg)', 0) / (h + 1e-5)
        row['Rain_per_Ha'] = row.get('30DRain( in mm)', 0) / (h + 1e-5)
        return row
    
    def predict(self, lp: float, dap: float, urea: float, pest: float,
                seed: float, hectares: float, rain: float) -> dict:
        """
        Predict paddy yield based on input parameters.
        
        Args:
            lp: Fertilizer in nursery area (Tonnes)
            dap: DAP at 20 days (Kg)
            urea: Urea at 40 days (Kg)
            pest: Pesticide at 60 days (ml)
            seed: Seed quantity (Kg)
            hectares: Field area (Hectares)
            rain: Rainfall first 30 days (mm)
            
        Returns:
            Dictionary containing:
                - prediction: Total yield in Kg
                - per_hectare: Yield per hectare
                - tonnes: Yield in tonnes
                - bags: Approximate 50kg bags
                - hectares: Input hectare value
                - summary: Model performance summary
        """
        # Build input row
        row = {
            'LP_nurseryarea(in Tonnes)': float(lp),
            'DAP_20days': float(dap),
            'Urea_40Days': float(urea),
            'Pest_60Day(in ml)': float(pest),
            'Seedrate(in Kg)': float(seed),
            'Hectares': float(hectares),
            '30DRain( in mm)': float(rain)
        }
        
        # Engineer features
        row = self.engineer_features(row)
        
        # Prepare for prediction
        X_input = pd.DataFrame([row])[self.features]
        X_scaled = self.scaler.transform(X_input)
        
        # Predict
        prediction = float(self.model.predict(X_scaled)[0])
        
        # Calculate derived metrics
        result = {
            'prediction': prediction,
            'per_hectare': prediction / max(float(hectares), 0.1),
            'tonnes': prediction / 1000,
            'bags': int(prediction / 50),
            'hectares': hectares,
            'summary': self.summary
        }
        
        return result
    
    @staticmethod
    def get_tip(prediction: float, hectares: float) -> str:
        """Get simple rule-based tip based on predicted yield."""
        prediction / max(float(hectares), 0.1)
        
        if prediction < 20000:
            return '⚠️ Low yield. Adjust fertilizer/pesticide.'
        elif prediction > 42000:
            return '🌟 Excellent yield! Inputs well optimised.'
        else:
            return '👍 Good yield. Small DAP/pest tweaks may help.'
