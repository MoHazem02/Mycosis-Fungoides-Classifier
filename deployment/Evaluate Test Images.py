"""
Evaluate Test Images - Run MF Classifier against labeled test folders.

This script evaluates the classifier pipeline against test images organized in:
- G:/My Drive/CLPD-MF-Dataset/test file/MF/ (label: 1 = Mycosis Fungoides)
- G:/My Drive/CLPD-MF-Dataset/test file/NON-MF/
  - PL-Spectrum/ (label: 2 = PLEVA-PLC)
  - Pseudo-Lymphoma/ (label: 4 = pseudolymphoma)
  - T-Cell dyscrasia/ (label: 3 = T-cell dyscrasia)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import json
import sys
from datetime import datetime
import cv2
from PIL import Image
import hashlib
import gc

# Import from deployment modules
from classifier import MFClassifier
import config


# ── Configuration ────────────────────────────────────────────────────────────
TEST_DATA_ROOT = Path("G:/My Drive/CLPD-MF-Dataset/smart phone")

# Mapping of folder paths to ground truth labels
LABEL_MAPPING = {
    "MF": 1,                                    # Mycosis Fungoides
    "Non-MF/PL-Spectrum": 2,                    # PLEVA-PLC
    "Non-MF/pseudolymphoma": 4,                 # pseudolymphoma
    "Non-MF/T-Cell dyscrasia": 3,               # T-cell dyscrasia
    "Non-MF/other" : 3

}

CLASS_NAMES = config.CLASS_NAMES  # ["B cell Lymphoma", "Mycosis Fungoides", "PLEVA-PLC", "T-cell dyscrasia", "pseudolymphoma"]

def get_device() -> str:
    """Determine best device (cuda if available, else cpu)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def get_test_folders() -> dict:
    """
    Collect all patient folders from test data directories.
    Returns dict mapping folder_path -> ground_truth_label
    """
    test_folders = {}
    
    for folder_path_str, label in LABEL_MAPPING.items():
        folder_path = TEST_DATA_ROOT / folder_path_str
        
        if not folder_path.exists():
            print(f"⚠ Warning: {folder_path} does not exist")
            continue
        
        # Get all subdirectories (each is a patient folder)
        patient_folders = [d for d in folder_path.iterdir() if d.is_dir()]
        
        for patient_folder in patient_folders:
            test_folders[patient_folder] = label
            print(f"Found: {patient_folder.name:40s} → {CLASS_NAMES[label]}")
    
    return test_folders


def run_evaluation():
    """Main evaluation pipeline."""
    
    print("\n" + "="*80)
    print("MF Classifier - Test Set Evaluation")
    print("="*80 + "\n")
    
    # ── Setup ────────────────────────────────────────────────────────────
    device = get_device()
    print(f"\nTest data root: {TEST_DATA_ROOT}")
    print(f"Output classes: {CLASS_NAMES}\n")
    
    # ── Load test folders ────────────────────────────────────────────────
    print("Collecting test folders...")
    test_folders = get_test_folders()
    
    if not test_folders:
        print("❌ No test folders found!")
        return
    
    total_samples = len(test_folders)
    print(f"\n✓ Found {total_samples} test samples\n")
    
    # ── Initialize classifier ────────────────────────────────────────────
    print("Loading classifier...")
    classifier = MFClassifier(device=device)
    classifier.load_models()
    print("✓ Classifier loaded\n")
    
    # ── Run predictions ──────────────────────────────────────────────────
    print("Running predictions...")
    print("-" * 80)
    
    results = []
    ground_truths = []
    predictions = []
    predicted_classes = []
    confidences = []
    
    for idx, (patient_folder, true_label) in enumerate(test_folders.items(), 1):
        patient_name = patient_folder.name
        
        try:
            # Run prediction
            prediction_result = classifier.predict(patient_folder, is_smartphone=True)
            
            # Extract results
            pred_class_idx = prediction_result['predicted_class_idx']
            pred_class_name = prediction_result['predicted_class']
            confidence = prediction_result['confidence']
            class_probs = prediction_result['class_probabilities']
            
            # Store for metrics
            ground_truths.append(true_label)
            predictions.append(pred_class_idx)
            predicted_classes.append(pred_class_name)
            confidences.append(confidence)
            
            # Determine if correct (any non-MF prediction is correct for NON-MF samples)
            is_correct = pred_class_idx == true_label
            status = "✓" if is_correct else "✗"
            
            # Log result
            results.append({
                'patient_name': patient_name,
                'ground_truth_label': true_label,
                'ground_truth_class': CLASS_NAMES[true_label],
                'predicted_label': pred_class_idx,
                'predicted_class': pred_class_name,
                'confidence': confidence,
                'correct': is_correct,
                'class_probabilities': class_probs,
            })
            
            # Display progress
            true_class = CLASS_NAMES[true_label]
            print(f"[{idx:3d}/{total_samples}] {status} {patient_name:45s} | "
                  f"True: {true_class:20s} → Pred: {pred_class_name:20s} "
                  f"({confidence:.2%})")
            
        except Exception as e:
            print(f"[{idx:3d}/{total_samples}] ✗ {patient_name:45s} | Error: {str(e)}")
            results.append({
                'patient_name': patient_name,
                'ground_truth_label': true_label,
                'ground_truth_class': CLASS_NAMES[true_label],
                'error': str(e),
                'correct': False,
            })
            ground_truths.append(true_label)
            predictions.append(-1)  # Invalid prediction
    
    print("-" * 80)
    
    # ── Calculate metrics ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80 + "\n")
    
    # Overall accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"Overall Accuracy: {accuracy:.2%}\n")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, labels=list(range(len(CLASS_NAMES))), 
        zero_division=0
    )
    
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<25} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 80)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<25} {precision[i]:>12.3f} {recall[i]:>12.3f} "
              f"{f1[i]:>12.3f} {support[i]:>10.0f}")
    
    print("-" * 80)
    print(f"{'Weighted Avg':<25} {np.average(precision, weights=support):>12.3f} "
          f"{np.average(recall, weights=support):>12.3f} "
          f"{np.average(f1, weights=support):>12.3f} {np.sum(support):>10.0f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(ground_truths, predictions, labels=list(range(len(CLASS_NAMES))))
    
    print("Confusion Matrix:")
    print("-" * 80)
    print(f"{'Predicted →':<20}", end="")
    for class_name in CLASS_NAMES:
        print(f"{class_name[:15]:>15}", end="")
    print()
    
    for i, true_class in enumerate(CLASS_NAMES):
        print(f"{true_class[:20]:<20}", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i, j]:>15}", end="")
        print()
    
    print()
    
    # Binary (MF vs Non-MF) metrics
    print("\nBinary Classification (MF vs Non-MF):")
    print("-" * 80)
    
    # Convert to binary labels (1 for MF, 0 for Non-MF)
    binary_ground_truth = [1 if label == 1 else 0 for label in ground_truths]
    binary_predictions = [1 if pred == 1 else 0 for pred in predictions]
    
    binary_accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        binary_ground_truth, binary_predictions, average='binary', zero_division=0
    )
    
    print(f"Accuracy:  {binary_accuracy:.2%}")
    print(f"Precision: {binary_precision:.3f}")
    print(f"Recall:    {binary_recall:.3f}")
    print(f"F1-Score:  {binary_f1:.3f}\n")
    
    # ── Save Results to JSON ─────────────────────────────────────────────
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"test_evaluation_{timestamp}.json"
    
    # Prepare for JSON serialization
    json_results = []
    for result in results:
        result_copy = result.copy()
        # Convert numpy types to native Python types
        if 'confidence' in result_copy:
            result_copy['confidence'] = float(result_copy['confidence'])
        if 'class_probabilities' in result_copy:
            result_copy['class_probabilities'] = {
                k: float(v) for k, v in result_copy['class_probabilities'].items()
            }
        json_results.append(result_copy)
    
    summary = {
        'timestamp': timestamp,
        'total_samples': total_samples,
        'accuracy': float(accuracy),
        'binary_accuracy': float(binary_accuracy),
        'binary_precision': float(binary_precision),
        'binary_recall': float(binary_recall),
        'binary_f1': float(binary_f1),
        'per_class_precision': {CLASS_NAMES[i]: float(precision[i]) for i in range(len(CLASS_NAMES))},
        'per_class_recall': {CLASS_NAMES[i]: float(recall[i]) for i in range(len(CLASS_NAMES))},
        'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(len(CLASS_NAMES))},
        'confusion_matrix': cm.tolist(),
        'device_used': device,
    }
    
    evaluation_data = {
        'summary': summary,
        'results': json_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # ── Save Summary to CSV ──────────────────────────────────────────────
    csv_file = output_dir / f"test_evaluation_summary_{timestamp}.csv"
    
    summary_data = []
    for result in results:
        row = {
            'patient_name': result.get('patient_name', ''),
            'ground_truth': result.get('ground_truth_class', ''),
            'predicted': result.get('predicted_class', ''),
            'confidence': result.get('confidence', 0),
            'correct': result.get('correct', False),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    print(f"✓ Summary saved to: {csv_file}\n")
    
    print("="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    run_evaluation()
