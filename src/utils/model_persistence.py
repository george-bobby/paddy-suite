"""
Model Persistence — Unified utilities for saving/loading ML models and artifacts
Eliminates duplicate save/load logic across training modules
"""
import json
import pickle
import joblib
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union


class ModelArtifacts:
    """
    Standardized model artifact management for ML models.
    
    Handles saving/loading of models, scalers, encoders, and metadata
    with consistent file structure and error handling.
    """
    
    @staticmethod
    def save_sklearn_model(
        model: Any,
        scaler: Optional[Any],
        metadata: Dict[str, Any],
        base_path: Union[str, Path],
        **additional_artifacts
    ) -> None:
        """
        Save a scikit-learn style model with artifacts.
        
        Args:
            model: Trained model (sklearn, xgboost, lightgbm, etc.)
            scaler: Optional data scaler/transformer
            metadata: Dictionary of metrics and model info (saved as JSON)
            base_path: Directory to save artifacts
            **additional_artifacts: Additional objects to save (e.g., label_encoders, features)
        
        Standard file structure:
            base_path/
                model.pkl         - The trained model
                scaler.pkl        - Data scaler (if provided)
                metadata.json     - Model metrics and configuration
                {key}.pkl         - Any additional artifacts
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main model
            joblib.dump(model, base_path / 'model.pkl')
            print(f'  💾 Model saved → {base_path / "model.pkl"}')
            
            # Save scaler if provided
            if scaler is not None:
                joblib.dump(scaler, base_path / 'scaler.pkl')
                print(f'  💾 Scaler saved → {base_path / "scaler.pkl"}')
            
            # Save metadata as JSON
            with open(base_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f'  💾 Metadata saved → {base_path / "metadata.json"}')
            
            # Save additional artifacts
            for key, artifact in additional_artifacts.items():
                artifact_path = base_path / f'{key}.pkl'
                joblib.dump(artifact, artifact_path)
                print(f'  💾 {key} saved → {artifact_path}')
                
        except Exception as e:
            raise RuntimeError(f"Failed to save model artifacts to {base_path}: {e}")
    
    @staticmethod
    def load_sklearn_model(
        base_path: Union[str, Path],
        load_scaler: bool = True
    ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        """
        Load a scikit-learn style model with artifacts.
        
        Args:
            base_path: Directory containing saved artifacts
            load_scaler: Whether to load scaler.pkl (if exists)
            
        Returns:
            Tuple of (model, scaler, metadata)
            scaler is None if not found or load_scaler=False
            
        Raises:
            FileNotFoundError: If required files missing
            RuntimeError: If loading fails
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {base_path}\n"
                f"Train the model first before loading."
            )
        
        try:
            # Load model
            model_path = base_path / 'model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = joblib.load(model_path)
            
            # Load scaler if requested and exists
            scaler = None
            if load_scaler:
                scaler_path = base_path / 'scaler.pkl'
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata = {}
            metadata_path = base_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return model, scaler, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {base_path}: {e}")
    
    @staticmethod
    def load_artifact(base_path: Union[str, Path], artifact_name: str) -> Any:
        """
        Load a specific artifact by name.
        
        Args:
            base_path: Model directory
            artifact_name: Name of artifact (without .pkl extension)
            
        Returns:
            The loaded artifact
        """
        artifact_path = Path(base_path) / f'{artifact_name}.pkl'
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        return joblib.load(artifact_path)
    
    @staticmethod
    def save_pytorch_model(
        model: torch.nn.Module,
        metadata: Dict[str, Any],
        base_path: Union[str, Path],
        save_full_model: bool = False,
        **additional_artifacts
    ) -> None:
        """
        Save a PyTorch model with artifacts.
        
        Args:
            model: Trained PyTorch model
            metadata: Dictionary of metrics and model info
            base_path: Directory to save artifacts
            save_full_model: If True, saves entire model; else just state_dict (recommended)
            **additional_artifacts: Additional objects (saved as pickle)
        
        Standard file structure:
            base_path/
                best.pth          - Model state_dict or full model
                config.json       - Model configuration and metadata
                {key}.pkl         - Additional artifacts
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model_path = base_path / 'best.pth'
            if save_full_model:
                torch.save(model, model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f'  💾 PyTorch model saved → {model_path}')
            
            # Save metadata
            config_path = base_path / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f'  💾 Config saved → {config_path}')
            
            # Save additional artifacts
            for key, artifact in additional_artifacts.items():
                if key.endswith('_encoder') or key.endswith('_le'):
                    # Use pickle for encoders
                    artifact_path = base_path / f'{key}.pkl'
                    with open(artifact_path, 'wb') as f:
                        pickle.dump(artifact, f)
                else:
                    artifact_path = base_path / f'{key}.pkl'
                    joblib.dump(artifact, artifact_path)
                print(f'  💾 {key} saved → {artifact_path}')
                
        except Exception as e:
            raise RuntimeError(f"Failed to save PyTorch model to {base_path}: {e}")
    
    @staticmethod
    def load_pytorch_model(
        model_class: torch.nn.Module,
        base_path: Union[str, Path],
        device: str = 'cpu',
        load_full_model: bool = False
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load a PyTorch model with config.
        
        Args:
            model_class: Initialized model (for state_dict loading) or None (for full model)
            base_path: Directory containing saved model
            device: Device to load model on ('cpu' or 'cuda')
            load_full_model: If True, loads entire model; else loads state_dict
            
        Returns:
            Tuple of (model, config)
        """
        base_path = Path(base_path)
        model_path = base_path / 'best.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load config
            config = {}
            config_path = base_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Load model
            if load_full_model:
                model = torch.load(model_path, map_location=device)
            else:
                if model_class is None:
                    raise ValueError("model_class required when load_full_model=False")
                model = model_class
                model.load_state_dict(torch.load(model_path, map_location=device))
            
            model = model.to(device)
            model.eval()
            
            return model, config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {base_path}: {e}")


# Convenience functions
def save_model(model: Any, base_path: Union[str, Path], **kwargs) -> None:
    """Auto-detect model type and save appropriately."""
    if isinstance(model, torch.nn.Module):
        ModelArtifacts.save_pytorch_model(model, base_path=base_path, **kwargs)
    else:
        ModelArtifacts.save_sklearn_model(model, base_path=base_path, **kwargs)


def load_model(base_path: Union[str, Path], **kwargs) -> Any:
    """Load model (auto-detect type from file extension)."""
    base_path = Path(base_path)
    if (base_path / 'best.pth').exists():
        return ModelArtifacts.load_pytorch_model(base_path=base_path, **kwargs)
    else:
        return ModelArtifacts.load_sklearn_model(base_path=base_path, **kwargs)
