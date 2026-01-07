# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MF Classifier
Build command: pyinstaller build.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Get the deployment directory
DEPLOYMENT_DIR = Path(SPECPATH)

a = Analysis(
    ['app.py'],
    pathex=[str(DEPLOYMENT_DIR)],
    binaries=[],
    datas=[
        # Include model files
        ('models/model_resnet50_x10.pth', 'models'),
        ('models/model_resnet50_x20.pth', 'models'),
        # Include logo images
        ('assets/CUFE.png', 'assets'),
        ('assets/Kasr.png', 'assets'),
        # Include config
        ('config.py', '.'),
        ('classifier.py', '.'),
        ('preprocessing.py', '.'),
        ('report_generator.py', '.'),
    ],
    hiddenimports=[
        # Timm models
        'timm',
        'timm.models',
        'timm.models.resnet',
        'timm.models.layers',
        'timm.data',
        # Torch
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchvision.transforms',
        # CustomTkinter
        'customtkinter',
        # ReportLab
        'reportlab',
        'reportlab.lib',
        'reportlab.platypus',
        'reportlab.lib.pagesizes',
        'reportlab.lib.styles',
        'reportlab.lib.colors',
        'reportlab.lib.units',
        # PIL
        'PIL',
        'PIL.Image',
        # OpenCV
        'cv2',
        # Numpy
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn',
        'jupyter',
        'notebook',
        'IPython',
        'tkinter.test',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MF_Classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)
