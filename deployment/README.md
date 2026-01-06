# Mycosis Fungoides Classifier - Deployment

Desktop application for classifying Mycosis Fungoides from histopathology images using late fusion of x10 and x20 magnification models.

## Features

- **CPU-only inference** for maximum portability
- **Late fusion** of x10 and x20 magnification predictions
- **PDF report generation** with patient details
- **Modern dark UI** using CustomTkinter

## Prerequisites

1. Python 3.10+ (for building)
2. Trained model files:
   - `model_resnet50_x10.pth`
   - `model_resnet50_x20.pth`

## Setup for Development

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Copy model files from training
copy "..\Saved Models\Top Performer\model_resnet50_x10.pth" "models\"
copy "..\Saved Models\Top Performer\model_resnet50_x20.pth" "models\"
```

## Run in Development Mode

```powershell
python app.py
```

## Build Executable

```powershell
# Ensure models are in place
dir models\

# Build with PyInstaller
pyinstaller build.spec

# Output will be in dist\MF_Classifier.exe
```

## Usage

1. Launch `MF_Classifier.exe`
2. Click "Select Patient Folder"
3. Choose a folder with this structure:
   ```
   PatientName/
   ├── x10/
   │   ├── image1.tif
   │   └── image2.tif
   └── x20/
       ├── image1.tif
       └── image2.tif
   ```
4. Click "Analyze Patient"
5. View results and optionally save PDF report

## Configuration

Edit `config.py` to modify:

- `OPTIMAL_FUSION_WEIGHT`: Late fusion weight (from Brier score optimization)
- `PATCH_SIZE`, `PATCH_STRIDE`: Patch extraction parameters
- `BATCH_SIZE`: Inference batch size

## File Structure

```
deployment/
├── app.py              # Main GUI application
├── classifier.py       # Model inference logic
├── preprocessing.py    # Patch extraction
├── config.py           # Configuration constants
├── report_generator.py # PDF report generation
├── requirements.txt    # Python dependencies
├── build.spec          # PyInstaller configuration
├── models/             # Trained model weights
│   ├── model_resnet50_x10.pth
│   └── model_resnet50_x20.pth
└── assets/
    └── icon.ico        # Application icon (optional)
```

## Troubleshooting

### "Models not found" error
Ensure model files are copied to `models/` folder.

### Slow first launch
The application loads ~200MB of model weights on startup. Subsequent predictions are faster.

### Build size too large
The CPU-only PyTorch build should be ~400-500MB. If larger, check that CUDA packages are not included.
