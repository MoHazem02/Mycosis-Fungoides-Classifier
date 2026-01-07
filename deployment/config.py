"""
Configuration settings for MF Classifier deployment.
"""

# Model Architecture
MODEL_ARCHITECTURE = "resnet50"
NUM_CLASSES = 2
CLASS_NAMES = ["MF", "Non-MF"]

# Model hyperparameters (from training)
DROPOUT_X10 = 0.3979527105843245
DROPOUT_X20 = 0.2582534390245415

# Late Fusion Settings
# Optimal weight from Brier Score minimization
# p_fused = w * p_x10 + (1 - w) * p_x20
OPTIMAL_FUSION_WEIGHT = 0.65  

# Image Processing
PATCH_SIZE = 512
PATCH_STRIDE = 256
MIN_FOREGROUND_RATIO = 0.285
MAX_PATCHES_PER_IMAGE = 200
IMAGE_SIZE = (512, 512)

# Normalization (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Inference
BATCH_SIZE = 8  # Smaller batch for CPU inference
CLASSIFICATION_THRESHOLD = 0.5

# Paths (relative to deployment folder)
MODEL_X10_FILENAME = "model_resnet50_x10.pth"
MODEL_X20_FILENAME = "model_resnet50_x20.pth"

# PDF Report Settings
PDF_TITLE = "AI-Assisted Mycosis Fungoides Classification Report"
PDF_INSTITUTION = ("Faculty of Engineering, Cairo University <br/>"
                          "Department of Dermatology, Kasr Al-Ainy Hospitals")
