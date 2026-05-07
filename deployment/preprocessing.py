"""
Preprocessing utilities for patch extraction from histology images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import config


def extract_patches_from_image(
    img_path: Path,
    patch_size: int,
    stride: int,
    min_foreground_ratio: float = config.MIN_FOREGROUND_RATIO,
    max_patches: int = config.MAX_PATCHES_PER_IMAGE
) -> List[np.ndarray]:
    """
    Extract tissue patches from a histology image.
    
    Args:
        img_path: Path to the image file
        patch_size: Size of each square patch
        stride: Stride between patches
        min_foreground_ratio: Minimum tissue content required
        max_patches: Maximum number of patches to extract
        
    Returns:
        List of patch arrays (RGB, uint8)
    """
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    patches = []
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            crop = img.crop((x, y, x + patch_size, y + patch_size))
            arr = np.asarray(crop)
            
            # Convert to HSV and check saturation for tissue detection
            hsv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            saturation = hsv_img[:, :, 1]
            fg_ratio = (saturation > 20).mean()
            
            if fg_ratio < min_foreground_ratio: continue
            
            patches.append(arr)
            
            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break
    
    return patches

def extract_patches_from_folder(
    folder_path: Path,
    mag: int,
    progress_callback=None
) -> Tuple[List[np.ndarray], int]:
    """
    Extract patches from all images in a magnification folder.
    
    Args:
        folder_path: Path to folder containing .tif images
        mag: Magnification level (10 or 20)
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        Tuple of (list of patches, number of images processed)
    """
    all_patches = []
    
    # Find all .tif images or jpg/png if no .tif found
    image_files = list(folder_path.glob('*.tif'))
    if not image_files:
        # Also check for other formats
        image_files = list(folder_path.glob('*.tiff')) + \
                      list(folder_path.glob('*.png')) + \
                      list(folder_path.glob('*.jpg')) + \
                      list(folder_path.glob('*.jpeg'))
    
    total_images = len(image_files)
    
    for i, img_path in enumerate(image_files):
        if mag == 10:
            patches = extract_patches_from_image(img_path, patch_size=config.PATCH_SIZE_10x, stride=config.PATCH_STRIDE_10x)
        else:
            patches = extract_patches_from_image(img_path, patch_size=config.PATCH_SIZE_20x, stride=config.PATCH_STRIDE_20x)
        all_patches.extend(patches)
        
        if progress_callback:
            progress_callback(i + 1, total_images)
    
    return all_patches, total_images


def validate_patient_folder(folder_path: Path) -> Tuple[bool, str]:
    """
    Validate that a patient folder has the correct structure.
    
    Args:
        folder_path: Path to patient folder
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        return False, "Folder does not exist"
    
    if not folder_path.is_dir():
        return False, "Path is not a directory"
    
    x10_folder = folder_path / "x10"
    x20_folder = folder_path / "x20"
    
    if not x10_folder.exists():
        return False, "Missing 'x10' subfolder"
    
    if not x20_folder.exists():
        return False, "Missing 'x20' subfolder"
    
    # Check for images
    x10_images = list(x10_folder.glob('*.tif')) + list(x10_folder.glob('*.tiff')) + list(x10_folder.glob('*.jpg')) + list(x10_folder.glob('*.jpeg')) + list(x10_folder.glob('*.png'))
    x20_images = list(x20_folder.glob('*.tif')) + list(x20_folder.glob('*.tiff')) + list(x20_folder.glob('*.jpg')) + list(x20_folder.glob('*.jpeg')) + list(x20_folder.glob('*.png'))
    
    if len(x10_images) == 0:
        return False, "No .tif or .jpg/.png images found in 'x10' folder"
    
    if len(x20_images) == 0:
        return False, "No .tif or .jpg/.png images found in 'x20' folder"
    
    return True, f"Found {len(x10_images)} x10 images and {len(x20_images)} x20 images"
