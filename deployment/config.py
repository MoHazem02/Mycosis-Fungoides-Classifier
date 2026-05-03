"""
Configuration settings for MF Classifier deployment.
"""

# Model Architecture
MODEL_ARCHITECTURE = "tf_efficientnet_b3"
NUM_CLASSES = 5
# Classes are sorted alphabetically by the dataset loader
CLASS_NAMES = ["B cell Lymphoma", "Mycosis Fungoides", "PLEVA-PLC", "T-cell dyscrasia", "pseudolymphoma"]

# Binary classification mapping
# MF = Class 1, Non-MF = Classes 0, 2, 3, 4
BINARY_CLASS_MAP = {
    'MF': 1,
    'Non-MF': [0, 2, 3, 4]
}

# Model hyperparameters 
DROPOUT_X10 = 0.4
DROPOUT_X20 = 0.4

# Late Fusion Settings
# Optimal weight from Brier Score minimization
# p_fused = w * p_x10 + (1 - w) * p_x20
OPTIMAL_FUSION_WEIGHT = 0.2 

# Optimal threshold for x20 MF binary classification, derived via Youden's J statistic on the ROC curve (training set).
X20_OPTIMAL_THRESHOLD: float = 0.8351 

# Image Processing
PATCH_SIZE_10x = 512
PATCH_SIZE_20x = 1024
PATCH_STRIDE_10x = 256
PATCH_STRIDE_20x = 512
MIN_FOREGROUND_RATIO = 0.285
MAX_PATCHES_PER_IMAGE = 200
IMAGE_SIZE = (512, 512)

# Normalization (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Inference
BATCH_SIZE = 8  # Smaller batch for CPU inference # TODO
CLASSIFICATION_THRESHOLD = 0.5 # TODO

# Paths (relative to deployment folder)
MODEL_X10_FILENAME = "model_tf_efficientnet_b3_x10_5class.pth"
MODEL_X20_FILENAME = "model_tf_efficientnet_b3_x20_5class.pth"

# PDF Report Settings
PDF_TITLE = "AI-Assisted Mycosis Fungoides Classification Report"
PDF_INSTITUTION = ("Faculty of Engineering, Cairo University <br/>"
                          "Department of Dermatology, Kasr Al-Ainy Hospitals")
