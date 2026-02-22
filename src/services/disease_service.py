"""
Disease Classification Service — Paddy disease identification from leaf images.
"""
import json
import pickle
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.cuda.amp import autocast
import timm
from PIL import Image

import config


class _SimpleLabelEncoder:
    """Minimal encoder-compatible container with a classes_ attribute."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)


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
        self.label_encoder = self._load_or_build_label_encoder()
        self.config = self._load_or_build_config()
        
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

    def _load_or_build_label_encoder(self):
        """Load encoder artifact, or rebuild it from disease CSV if missing."""
        enc_path = Path(config.DISEASE_ENCODER_PATH)
        if enc_path.exists():
            with enc_path.open('rb') as f:
                return pickle.load(f)

        classes = self._discover_classes()
        le = _SimpleLabelEncoder(classes)
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        with enc_path.open('wb') as f:
            pickle.dump(le, f)
        print(f'  ℹ️ Rebuilt missing disease artifact: {config.DISEASE_ENCODER_PATH}')
        return le

    def _load_or_build_config(self):
        """Load model config artifact, or regenerate sensible default if missing."""
        cfg_path = Path(config.DISEASE_CONFIG_PATH)
        if cfg_path.exists():
            with cfg_path.open() as f:
                return json.load(f)

        cfg = {
            'model_name': 'efficientnet_b3',
            'num_classes': len(self.label_encoder.classes_),
            'img_size': config.IMG2,
            'classes': list(self.label_encoder.classes_),
            'val_accuracy': None,
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open('w') as f:
            json.dump(cfg, f, indent=2)
        print(f'  ℹ️ Rebuilt missing disease artifact: {config.DISEASE_CONFIG_PATH}')
        return cfg

    def _discover_classes(self):
        """Discover disease classes from CSV first, then from train_images folders."""
        csv_path = Path(config.DISEASE_CSV)
        if csv_path.exists():
            with csv_path.open(newline='') as f:
                reader = csv.DictReader(f)
                if 'label' in reader.fieldnames:
                    labels = sorted({row['label'] for row in reader if row.get('label')})
                    if labels:
                        return labels

        train_dir = Path(config.DISEASE_TRAIN_DIR)
        if train_dir.exists():
            labels = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
            if labels:
                return labels

        raise FileNotFoundError(
            f"Cannot rebuild disease classes: missing {config.DISEASE_CSV} and "
            f"class folders under {config.DISEASE_TRAIN_DIR}. "
            "Run 'python train_disease.py' (or 'python main.py') once."
        )
    
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
