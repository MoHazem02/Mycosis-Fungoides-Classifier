"""
Helper script to load and use saved Random Forest models for real-time predictions.
Use this in another file or interactive session.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class MFClassifierLoader:
    """Wrapper to load and use saved MF prediction models."""
    
    def __init__(self, model_path_diagnosis=None, model_path_stage=None):
        """
        Load pre-trained models.
        
        Args:
            model_path_diagnosis: Path to rf_model_diagnosis_mf.pkl
            model_path_stage: Path to rf_model_stage_mf.pkl
        """
        self.diagnosis_bundle = None
        self.stage_bundle = None
        
        if model_path_diagnosis:
            self.diagnosis_bundle = joblib.load(model_path_diagnosis)
            print(f"✓ Loaded diagnosis model: {model_path_diagnosis}")
        
        if model_path_stage:
            self.stage_bundle = joblib.load(model_path_stage)
            print(f"✓ Loaded stage model: {model_path_stage}")
    
    def predict_diagnosis(self, X):
        """
        Predict MF vs Non-MF diagnosis.
        
        Args:
            X: pandas DataFrame or numpy array with feature columns
        
        Returns:
            predictions (labels): array of 'MF' or 'Non-MF'
            probabilities: array of prediction probabilities
        """
        if self.diagnosis_bundle is None:
            raise ValueError("Diagnosis model not loaded")
        
        bundle = self.diagnosis_bundle
        
        # Ensure X is DataFrame with correct columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=bundle['features'])
        
        # Select only the required features in the correct order
        X = X[bundle['features']]
        
        # Impute missing values
        X_imp = bundle['imputer'].transform(X)
        
        # Predict
        y_pred_enc = bundle['model'].predict(X_imp)
        y_prob = bundle['model'].predict_proba(X_imp)
        
        # Decode labels
        y_pred_labels = bundle['label_encoder'].inverse_transform(y_pred_enc)
        
        return y_pred_labels, y_prob
    
    def predict_stage(self, X):
        """
        Predict MF Stage (Patch-Plaque vs Tumor) — for MF patients only.
        
        Args:
            X: pandas DataFrame or numpy array with feature columns
        
        Returns:
            predictions (labels): array of 'Patch-Plaque' or 'Tumor'
            probabilities: array of prediction probabilities
        """
        if self.stage_bundle is None:
            raise ValueError("Stage model not loaded")
        
        bundle = self.stage_bundle
        
        # Ensure X is DataFrame with correct columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=bundle['features'])
        
        # Select only the required features in the correct order
        X = X[bundle['features']]
        
        # Impute missing values
        X_imp = bundle['imputer'].transform(X)
        
        # Predict
        y_pred_enc = bundle['model'].predict(X_imp)
        y_prob = bundle['model'].predict_proba(X_imp)
        
        # Decode labels
        y_pred_labels = bundle['label_encoder'].inverse_transform(y_pred_enc)
        
        return y_pred_labels, y_prob


# ── Example usage:
if __name__ == '__main__':
    
    # Initialize loader
    clf = MFClassifierLoader(
        model_path_diagnosis='rf_model_diagnosis_mf.pkl',
        model_path_stage='rf_model_stage_mf.pkl'
    )
    
    # Example: Create dummy patient data (16 features)
    # In practice, use real patient data with these columns:
    features = ['macules', 'bx1_morph', 'papules', 'age', 'color', 'patch',
                'bx2_morph', 'visit_type', 'duration_months', 'plaque', 'scales',
                'site_head_neck', 'site_ll', 'course', 'bx2_site', 'nodule']
    
    # Create sample patient (random values for demo)
    X_sample = pd.DataFrame([
        [1, 0, 1, 65, 0, 1, 2, 1, 24.0, 1, 1, 0, 1, 0, 1, 0]
    ], columns=features)
    
    print("\n=== Task 1: Diagnosis (MF vs Non-MF) ===")
    diag_pred, diag_prob = clf.predict_diagnosis(X_sample)
    print(f"Predicted: {diag_pred[0]}")
    print(f"Probabilities: {diag_prob[0]}")
    
    print("\n=== Task 2: Stage of MF (for MF patients) ===")
    stage_pred, stage_prob = clf.predict_stage(X_sample)
    print(f"Predicted: {stage_pred[0]}")
    print(f"Probabilities: {stage_prob[0]}")
