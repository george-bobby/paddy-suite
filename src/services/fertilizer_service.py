"""
Fertilizer Recommendation Service — Smart fertilizer type selection.
"""
import joblib
import numpy as np
import config


class FertilizerService:
    """Service for recommending optimal fertilizer type based on conditions."""
    
    def __init__(self):
        """Load trained fertilizer recommendation model and artifacts."""
        self.model = joblib.load(config.FERT_MODEL_PATH)
        self.scaler = joblib.load(config.FERT_SCALER_PATH)
        self.label_encoder = joblib.load(config.FERT_ENCODER_PATH)
        self.features = joblib.load(config.FERT_FEATURES_PATH)
    
    def predict(self, moisture: float, soil_temp: float, ec: float,
                air_temp: float, air_humidity: float, growth_phase: int) -> dict:
        """
        Recommend optimal fertilizer type based on environmental conditions.
        
        Args:
            moisture: Soil moisture (%)
            soil_temp: Soil temperature (°C)
            ec: Electrical conductivity (µS/cm)
            air_temp: Air temperature (°C)
            air_humidity: Air humidity (%)
            growth_phase: Growth phase (0=Vegetative, 1000=Reproductive)
            
        Returns:
            Dictionary containing:
                - recommendation: Recommended fertilizer type
                - confidence: Prediction confidence (0-100)
                - confidence_level: 'High', 'Medium', or 'Low'
                - probabilities: Dict mapping fertilizer types to probabilities
                - warning: Optional warning message if confidence is low
        """
        # Prepare input
        input_data = np.array([[
            moisture, soil_temp, ec, air_temp, air_humidity, growth_phase
        ]])
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction_encoded = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(probabilities[prediction_encoded] * 100)
        
        # Get all probabilities
        probabilities_dict = {
            self.label_encoder.classes_[i]: float(p * 100)
            for i, p in enumerate(probabilities)
        }
        
        # Sort by probability
        sorted_probabilities = dict(
            sorted(probabilities_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Determine confidence level
        if confidence >= 70:
            confidence_level = 'High'
        elif confidence >= 50:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        # Generate warning if needed
        warning = None
        if confidence < 60:
            warning = '⚠️ Low confidence — consult an agronomist.'
        
        return {
            'recommendation': prediction,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'probabilities': sorted_probabilities,
            'warning': warning
        }
    
    @staticmethod
    def get_confidence_icon(confidence: float) -> str:
        """Get emoji icon based on confidence level."""
        if confidence >= 70:
            return '🟢'
        elif confidence >= 50:
            return '🟡'
        else:
            return '🔴'
