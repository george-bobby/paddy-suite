"""
Irrigation Prediction Service — Smart irrigation decision support.
"""
import json
import joblib
import numpy as np
import config


class IrrigationService:
    """Service for predicting irrigation needs based on environmental conditions."""
    
    # Growth stage information
    GROWTH_STAGES = {
        (0, 20): ('Transplanting/Establishment (1–20d)', '🌱 Critical stage — consistent water needed.'),
        (21, 60): ('Tillering/Vegetative (21–60d)', '🌿 Maintain moisture for tillering.'),
        (61, 90): ('Panicle/Heading (61–90d)', '🌾 Water stress now reduces yield significantly.'),
        (91, 999): ('Grain Fill/Maturation (90+d)', '🍚 Water demand reduces at this stage.'),
    }
    
    def __init__(self):
        """Load trained irrigation prediction model and artifacts."""
        self.model = joblib.load(config.IRR_MODEL_PATH)
        self.scaler = joblib.load(config.IRR_SCALER_PATH)
        with open(config.IRR_METADATA_PATH) as f:
            self.metadata = json.load(f)
    
    def predict(self, crop_days: int, soil_moisture: float,
                temperature: float, humidity: float) -> dict:
        """
        Predict irrigation requirement based on environmental conditions.
        
        Args:
            crop_days: Days since transplanting
            soil_moisture: Soil moisture level (100-800)
            temperature: Air temperature (°C)
            humidity: Air humidity (%)
            
        Returns:
            Dictionary containing:
                - irrigation_needed: Boolean indicating if irrigation is needed
                - confidence: Prediction confidence (0-100)
                - irrigation_probability: Probability of needing irrigation
                - stage: Current growth stage
                - stage_note: Growth stage guidance
                - recommendations: List of specific recommendations
        """
        # Prepare input
        input_data = np.array([[crop_days, soil_moisture, temperature, humidity]])
        
        # Scale if needed
        if self.metadata.get('needs_scale', False):
            input_data = self.scaler.transform(input_data)
        
        # Predict
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Get growth stage
        stage, stage_note = self._get_growth_stage(crop_days)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            soil_moisture, temperature, humidity, stage_note
        )
        
        return {
            'irrigation_needed': bool(prediction == 1),
            'confidence': float(probabilities[prediction] * 100),
            'irrigation_probability': float(probabilities[1] * 100),
            'stage': stage,
            'stage_note': stage_note,
            'recommendations': recommendations,
            'metadata': self.metadata
        }
    
    def _get_growth_stage(self, crop_days: int) -> tuple:
        """Determine growth stage based on crop age."""
        for (min_days, max_days), (stage, note) in self.GROWTH_STAGES.items():
            if min_days <= crop_days <= max_days:
                return stage, note
        return 'Unknown Stage', 'Monitor conditions closely.'
    
    def _generate_recommendations(self, soil_moisture: float, temperature: float,
                                   humidity: float, stage_note: str) -> list:
        """Generate specific recommendations based on conditions."""
        recommendations = [stage_note]
        
        # Soil moisture recommendations
        if soil_moisture < 200:
            recommendations.append(
                f'💧 Soil moisture {soil_moisture} critically low — irrigate immediately.'
            )
        elif soil_moisture < 350:
            recommendations.append(
                f'💧 Soil moisture {soil_moisture} low — irrigation recommended.'
            )
        elif soil_moisture < 500:
            recommendations.append(
                f'💧 Soil moisture {soil_moisture} moderate — monitor closely.'
            )
        else:
            recommendations.append(f'💧 Soil moisture {soil_moisture} adequate.')
        
        # Temperature recommendations
        if temperature > 36:
            recommendations.append(
                f'🌡️ Temp {temperature}°C extreme — high evapotranspiration.'
            )
        elif temperature > 32:
            recommendations.append(
                f'🌡️ Temp {temperature}°C high — increased water demand.'
            )
        
        # Humidity recommendations
        if humidity < 20:
            recommendations.append(
                f'💨 Humidity {humidity}% very low — irrigate more frequently.'
            )
        elif humidity > 60:
            recommendations.append(
                f'💨 Humidity {humidity}% high — less irrigation may suffice.'
            )
        
        return recommendations
