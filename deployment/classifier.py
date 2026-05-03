"""
MF Classifier - Model loading and inference logic.
"""

import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from typing import List, Dict
import sys
import config
from preprocessing import extract_patches_from_folder, validate_patient_folder


def get_models_path() -> Path:
    """Get the path to models folder, works both in dev and bundled exe."""
    if getattr(sys, 'frozen', False):
        # Running as bundled exe
        base_path = Path(sys._MEIPASS)
    else:
        # Running in development
        base_path = Path(__file__).parent
    
    return base_path / "models"


class MFClassifier:
    """
    Mycosis Fungoides Classifier using late fusion of x10 and x20 models.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the classifier.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.models_path = get_models_path()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
        
        # Models will be loaded lazily
        self.model_x10 = None
        self.model_x20 = None
        self._models_loaded = False
    
    def _create_model(self, dropout: float) -> torch.nn.Module:
        """Create a model architecture."""
        model = timm.create_model(
            config.MODEL_ARCHITECTURE,
            pretrained=False,
            num_classes=config.NUM_CLASSES,
            drop_rate=dropout
        )
        return model
    
    def load_models(self, progress_callback=None) -> None:
        """
        Load both x10 and x20 models.
        
        Args:
            progress_callback: Optional callback(status_text) for progress updates
        """
        if self._models_loaded:
            return
        
        if progress_callback:
            progress_callback("Loading x10 model...")
        
        # Load x10 model
        self.model_x10 = self._create_model(config.DROPOUT_X10)
        x10_path = self.models_path / config.MODEL_X10_FILENAME
        self.model_x10.load_state_dict(torch.load(x10_path, map_location=self.device))
        self.model_x10.to(self.device)
        self.model_x10.eval()
        
        if progress_callback:
            progress_callback("Loading x20 model...")
        
        # Load x20 model
        self.model_x20 = self._create_model(config.DROPOUT_X20)
        x20_path = self.models_path / config.MODEL_X20_FILENAME
        self.model_x20.load_state_dict(torch.load(x20_path, map_location=self.device))
        self.model_x20.to(self.device)
        self.model_x20.eval()
        
        self._models_loaded = True
        
        if progress_callback:
            progress_callback("Models loaded successfully!")
    
    def _preprocess_patches(self, patches: List[np.ndarray]) -> torch.Tensor:
        """Convert patch arrays to tensor batch."""
        tensors = []
        for patch in patches:
            img = Image.fromarray(patch)
            tensor = self.transform(img)
            tensors.append(tensor)
        
        if not tensors:
            return torch.empty(0, 3, *config.IMAGE_SIZE)
        
        return torch.stack(tensors)
    
    def _run_inference(
        self, 
        model: torch.nn.Module, 
        patches: List[np.ndarray],
        progress_callback=None,
        prefix: str = ""
    ) -> np.ndarray:
        """
        Run inference on patches using a model.
        
        Returns:
            Array of shape (n_patches, n_classes) with probabilities for all classes
        """
        if not patches:
            return np.empty((0, config.NUM_CLASSES))
        
        all_probs = []
        batch_size = config.BATCH_SIZE
        total_batches = (len(patches) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size]
                batch_tensor = self._preprocess_patches(batch_patches).to(self.device)
                
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                # Store all class probabilities
                all_probs.extend(probs.cpu().numpy())
                
                if progress_callback:
                    batch_num = (i // batch_size) + 1
                    progress_callback(f"{prefix}Processing batch {batch_num}/{total_batches}")
        
        return np.array(all_probs)
    
    def _aggregate_to_patient_level(self, patch_probs: np.ndarray) -> np.ndarray:
        """
        Aggregate patch probabilities to patient level using mean.
        
        Args:
            patch_probs: Array of shape (n_patches, n_classes)
            
        Returns:
            Array of shape (n_classes,) with aggregated probabilities
        """
        if len(patch_probs) == 0:
            return np.ones(config.NUM_CLASSES) / config.NUM_CLASSES  # Uniform if no patches
        return np.mean(patch_probs, axis=0)
    
    def predict(
        self, 
        patient_folder: Path,
        progress_callback=None
    ) -> Dict:
        """
        Predict MF/Non-MF for a patient folder.
        
        Args:
            patient_folder: Path to patient folder with x10/ and x20/ subfolders
            progress_callback: Optional callback(status_text) for progress updates
            
        Returns:
            Dictionary with prediction results
        """
        patient_folder = Path(patient_folder)
        patient_name = patient_folder.name
        
        # Validate folder structure
        is_valid, message = validate_patient_folder(patient_folder)
        if not is_valid:
            raise ValueError(f"Invalid patient folder: {message}")
        
        # Ensure models are loaded
        self.load_models(progress_callback)
        
        # Extract patches from x10
        if progress_callback:
            progress_callback("Extracting x10 patches...")
        x10_folder = patient_folder / "x10"
        x10_patches, n_x10_images = extract_patches_from_folder(x10_folder, mag=10)
        
        # Extract patches from x20
        if progress_callback:
            progress_callback("Extracting x20 patches...")
        x20_folder = patient_folder / "x20"
        x20_patches, n_x20_images = extract_patches_from_folder(x20_folder, mag=20)
        
        # Run inference on x10 patches
        if progress_callback:
            progress_callback("Running x10 model inference...")
        x10_patch_probs = self._run_inference(
            self.model_x10, x10_patches, progress_callback, "x10: "
        )
        
        # Run inference on x20 patches
        if progress_callback:
            progress_callback("Running x20 model inference...")
        x20_patch_probs = self._run_inference(
            self.model_x20, x20_patches, progress_callback, "x20: "
        )
        
        # Aggregate to patient level
        x10_patient_prob = self._aggregate_to_patient_level(x10_patch_probs)
        x20_patient_prob = self._aggregate_to_patient_level(x20_patch_probs)
        
        # Late fusion - weighted average of class probabilities
        w = config.OPTIMAL_FUSION_WEIGHT
        fused_probs = w * x10_patient_prob + (1 - w) * x20_patient_prob
        
        # Classification - get class with highest probability
        predicted_class_idx = int(np.argmax(fused_probs))
        predicted_class = config.CLASS_NAMES[predicted_class_idx]
        confidence = float(fused_probs[predicted_class_idx])
        
        # Binary classification: MF vs Non-MF
        mf_prob = float(fused_probs[1])  # Class 1 = MF (Mycosis Fungoides)
        non_mf_prob = float(fused_probs[0] + fused_probs[2] + fused_probs[3] + fused_probs[4])  # Sum of all Non-MF classes
        is_mf = mf_prob > non_mf_prob
        binary_prediction = 'MF' if is_mf else 'Non-MF'
        binary_confidence = max(mf_prob, non_mf_prob)
        
        # Build class probabilities dictionary
        class_probs = {
            config.CLASS_NAMES[i]: float(fused_probs[i]) 
            for i in range(config.NUM_CLASSES)
        }
        
        results = {
            'patient_name': patient_name,
            # Binary level
            'binary_prediction': binary_prediction,
            'binary_confidence': binary_confidence,
            'mf_probability': mf_prob,
            'non_mf_probability': non_mf_prob,
            # Multi-class level
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'x10_probabilities': {
                config.CLASS_NAMES[i]: float(x10_patient_prob[i])
                for i in range(config.NUM_CLASSES)
            },
            'x20_probabilities': {
                config.CLASS_NAMES[i]: float(x20_patient_prob[i])
                for i in range(config.NUM_CLASSES)
            },
            'fusion_weight': w,
            'n_x10_images': n_x10_images,
            'n_x20_images': n_x20_images,
            'n_x10_patches': len(x10_patches),
            'n_x20_patches': len(x20_patches)
        }
        
        if progress_callback:
            progress_callback("Prediction complete!")
        
        return results
