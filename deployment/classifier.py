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
    
    def _create_model(self, dropout: float = 0.4) -> torch.nn.Module:
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
        self.model_x10 = self._create_model()
        x10_path = self.models_path / config.MODEL_X10_FILENAME
        self.model_x10.load_state_dict(torch.load(x10_path, map_location=self.device))
        self.model_x10.to(self.device)
        self.model_x10.eval()
        
        if progress_callback:
            progress_callback("Loading x20 model...")
        
        # Load x20 model
        self.model_x20 = self._create_model()
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
        batch_size: int,
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

        # ── Extract patches ───────────────────────────────────────────────────────
        if progress_callback:
            progress_callback("Extracting x10 patches...")
        x10_patches, n_x10_images = extract_patches_from_folder(patient_folder / "x10", mag=10)
        if progress_callback:
            progress_callback("Extracting x20 patches...")
        x20_patches, n_x20_images = extract_patches_from_folder(patient_folder / "x20", mag=20)
        # ── Run inference ─────────────────────────────────────────────────────────
        if progress_callback:
            progress_callback("Running x10 model inference...")
        x10_patch_probs = self._run_inference(
            self.model_x10, x10_patches, config.BATCH_SIZE_X10, progress_callback, "x10: "
        )

        if progress_callback:
            progress_callback("Running x20 model inference...")
        x20_patch_probs = self._run_inference(
            self.model_x20, x20_patches, config.BATCH_SIZE_X20, progress_callback, "x20: "
        )

        # ── Patient-level aggregation (Mean across patches) ──────────────────────────
        x10_patient_prob = self._aggregate_to_patient_level(x10_patch_probs) # Shape: (5,)
        x20_patient_prob = self._aggregate_to_patient_level(x20_patch_probs) # Shape: (5,)

       # ── Step 1: Extract Non-MF Probabilities (Matching Notebook Logic) ───────────
        # Note: Binary Labels: MF = 0, Non-MF = 1. We track the Non-MF probability.
        mf_idx = 1
        p10_non_mf = 1.0 - float(x10_patient_prob[mf_idx])
        p20_non_mf = 1.0 - float(x20_patient_prob[mf_idx])

        
        # We stretch/squish the x20 scale so its 0.8351 boundary shifts to a standard 0.5
        if p20_non_mf < config.X20_OPTIMAL_THRESHOLD:
            p20_non_mf_cal = p20_non_mf * (0.5 / config.X20_OPTIMAL_THRESHOLD)
        else:
            p20_non_mf_cal = 0.5 + ((p20_non_mf - config.X20_OPTIMAL_THRESHOLD) * (0.5 / (1.0 - config.X20_OPTIMAL_THRESHOLD)))

        # ── Step 3: Soft-Voting Fusion ───────────────────────────────────────────────
        w_10 = config.OPTIMAL_FUSION_WEIGHT  # 0.20 for x10
        w_20 = 1.0 - w_10                    # 0.80 for x20
        
        fused_non_mf_prob = (w_10 * p10_non_mf) + (w_20 * p20_non_mf_cal)
        fused_mf_prob = 1.0 - fused_non_mf_prob

        # ── Step 4: Final Binary Call ────────────────────────────────────────────────
        # Because we calibrated x20 to a 0.5 center, anything > 0.5 is a Mimic (Non-MF).
        is_non_mf = fused_non_mf_prob > 0.5 
        
        binary_prediction = "Non-MF" if is_non_mf else "MF"
        binary_confidence = fused_non_mf_prob if is_non_mf else fused_mf_prob

        # ── Step 5: Multi-class Calibration and Fusion ───────────────────────────────
        
        # 1. Calculate the Scaling Ratio for the mimics
        # (Avoid division by zero just in case the model is 100% sure it's MF)
        mimic_scale_ratio = p20_non_mf_cal / p20_non_mf if p20_non_mf > 0 else 1.0
        
        # 2. Build the fully calibrated x20 array
        x20_patient_prob_calibrated = np.copy(x20_patient_prob)
        for i in range(config.NUM_CLASSES):
            if i == mf_idx:
                # MF gets the exact calibrated inverse
                x20_patient_prob_calibrated[i] = 1.0 - p20_non_mf_cal
            else:
                # Individual mimics are scaled proportionally
                x20_patient_prob_calibrated[i] = x20_patient_prob[i] * mimic_scale_ratio

        # 3. Fuse using the CALIBRATED x20 array
        calibrated_fused_array = (w_10 * x10_patient_prob) + (w_20 * x20_patient_prob_calibrated)
        
        if not is_non_mf:
            predicted_class_idx = mf_idx
            predicted_class = "MF"
            confidence = fused_mf_prob
        else:
            # Mask out MF to force a mimic choice
            masked_array = np.copy(calibrated_fused_array)
            masked_array[mf_idx] = -1.0 
            
            predicted_class_idx = int(np.argmax(masked_array))
            predicted_class = config.CLASS_NAMES[predicted_class_idx]
            confidence = fused_non_mf_prob 


        # ── Step 6: Construct the Traceability Dictionary ────────────────────────────
        results = {
            "patient_name":           patient_name,
            
            # Binary
            "binary_prediction":      binary_prediction,
            "binary_confidence":      float(binary_confidence),
            "mf_probability":         float(fused_mf_prob),
            "non_mf_probability":     float(fused_non_mf_prob),
            
            # Decision traceability
            "x10_raw_mf_prob":        float(x10_patient_prob[mf_idx]),
            "x20_raw_mf_prob":        float(x20_patient_prob[mf_idx]),
            "x20_calibrated_non_mf":  float(p20_non_mf_cal), 
            "x10_decision":           "MF" if p10_non_mf <= config.X10_OPTIMAL_THRESHOLD else "Non-MF",
            "x20_decision":           "MF" if p20_non_mf <= config.X20_OPTIMAL_THRESHOLD else "Non-MF",
            "x10_threshold":          config.X10_OPTIMAL_THRESHOLD,
            "x20_optimal_threshold":  config.X20_OPTIMAL_THRESHOLD,
            
            # Multi-class
            "predicted_class":        predicted_class,
            "predicted_class_idx":    predicted_class_idx,
            "confidence":             float(confidence),
            # Notice we now return the mathematically consistent calibrated array!
            "class_probabilities":    {config.CLASS_NAMES[i]: float(calibrated_fused_array[i]) for i in range(config.NUM_CLASSES)},
            "x10_probabilities":      {config.CLASS_NAMES[i]: float(x10_patient_prob[i]) for i in range(config.NUM_CLASSES)},
            "x20_probabilities":      {config.CLASS_NAMES[i]: float(x20_patient_prob_calibrated[i]) for i in range(config.NUM_CLASSES)},
            
            # Metadata
            "fusion_weight_x10":      w_10,
            "n_x10_images":           n_x10_images,
            "n_x20_images":           n_x20_images,
            "n_x10_patches":          len(x10_patches),
            "n_x20_patches":          len(x20_patches),
        }
        if progress_callback:
            progress_callback("Prediction complete!")

        return results