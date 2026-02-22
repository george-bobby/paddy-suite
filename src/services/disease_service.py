"""
Disease Classification Service — Paddy disease identification from leaf images.
"""
import json
import pickle
import torch
import torchvision.transforms as T
from torch.cuda.amp import autocast
import timm
from PIL import Image
import config


class DiseaseService:
    """Service for classifying paddy diseases from leaf images using EfficientNet-B3."""
    
    # Disease information database
    DISEASE_INFO = {
        'bacterial_leaf_blight': ('💧', 'Water-soaked to yellowish stripe on leaf margins.'),
        'bacterial_leaf_streak': ('💧', 'Dark brown streaks with wavy margins on leaves.'),
        'bacterial_panicle_blight': ('💧', 'Grain discoloration & panicle sterility.'),
        'blast': ('🍄', 'Diamond-shaped lesions with gray centers.'),
        'brown_spot': ('🟤', 'Circular brown spots with yellow halo.'),
        'dead_heart': ('☠️', 'Central shoot dies — caused by stem borers.'),
        'downy_mildew': ('🌫️', 'Yellowish patches, white fungal growth below.'),
        'hispa': ('🐛', 'White blotches caused by leaf mining larvae.'),
        'normal': ('✅', 'Healthy paddy plant — no disease detected.'),
        'tungro': ('🟡', 'Yellow-orange discoloration, stunted growth.'),
    }
    
    def __init__(self):
        """Load trained disease classification model and artifacts."""
        # Load config
        with open(config.DISEASE_CONFIG_PATH) as f:
            self.config = json.load(f)
        
        # Load label encoder
        with open(config.DISEASE_ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load model
        self.model = timm.create_model(
            'efficientnet_b3',
            pretrained=False,
            num_classes=self.config['num_classes']
        )
        self.model.load_state_dict(
            torch.load(config.DISEASE_MODEL_PATH, map_location=config.DEVICE)
        )
        self.model = self.model.to(config.DEVICE)
        self.model.eval()
        
        # Prepare transform
        self.transform = T.Compose([
            T.Resize((self.config['img_size'], self.config['img_size'])),
            T.ToTensor(),
            T.Normalize(config.IMG_MEAN, config.IMG_STD),
        ])
    
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """
        Classify paddy disease from leaf image.
        
        Args:
            image: PIL Image of paddy leaf
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing:
                - top_class: Most likely disease class
                - emoji: Emoji representing the disease
                - description: Description of the disease
                - probabilities: List of (class, probability, emoji, description) tuples
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(config.DEVICE)
        
        # Predict
        with torch.no_grad(), autocast():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = probabilities.topk(top_k)
        
        # Get top class info
        top_class = self.label_encoder.classes_[top_indices[0].item()]
        emoji, description = self.DISEASE_INFO.get(top_class, ('🌾', ''))
        
        # Build results for all top-k
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.label_encoder.classes_[idx.item()]
            class_emoji, class_desc = self.DISEASE_INFO.get(class_name, ('🌾', ''))
            predictions.append({
                'class': class_name,
                'probability': float(prob.item()),
                'emoji': class_emoji,
                'description': class_desc
            })
        
        return {
            'top_class': top_class,
            'emoji': emoji,
            'description': description,
            'predictions': predictions
        }
