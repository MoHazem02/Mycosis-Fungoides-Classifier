"""
Preprocessing utilities for patch extraction from histology images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import config


def smart_crop_microscope(image_path):
    """
    Performs a 'smart crop' on a microscope image to remove large black borders
    and isolate the circular illuminated field of view.
    """
    # 1. Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Method 1: Contour Detection
    # Convert to grayscale and apply slight Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary threshold
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_rgb
        
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the original image to this bounding box
    cropped = img_rgb[y:y+h, x:x+w].copy()
    
    # 3. Method 2: Circular Masking (Cleanup)
    # Create a circular mask matching the cropped bounding box
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    # The radius should fit within the bounding box
    radius = min(w // 2, h // 2)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply the mask to the cropped image, converting remaining black corners to white
    # Create a white background image
    white_bg = np.ones_like(cropped) * 255
    
    # Where mask is 255, use the cropped image, else use white background
    final_img = np.where(mask[..., None] == 255, cropped, white_bg)
    
    return final_img


def extract_patches_from_image(
    img_path: Path,
    patch_size: int,
    stride: int,
    min_foreground_ratio: float = config.MIN_FOREGROUND_RATIO,
    max_patches: int = config.MAX_PATCHES_PER_IMAGE,
    is_smartphone: bool = False
) -> List[np.ndarray]:
    """
    Extract tissue patches from a histology image.
    If is_smartphone is True, applies smart crop preprocessing and uses ratio 0.7.
    
    Args:
        img_path: Path to the image file
        patch_size: Size of each square patch
        stride: Stride between patches
        min_foreground_ratio: Minimum tissue content required
        max_patches: Maximum number of patches to extract
        is_smartphone: Whether to use smartphone specific preprocessing
        
    Returns:
        List of patch arrays (RGB, uint8)
    """
    if is_smartphone:
        try:
            cropped_array = smart_crop_microscope(img_path)
            img = Image.fromarray(cropped_array).convert('RGB')
        except Exception as e:
            print(f"Warning: Smart crop failed for {img_path}, using original image: {e}")
            img = Image.open(img_path).convert('RGB')
        
        min_foreground_ratio = 0.7
    else:
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
    progress_callback=None,
    is_smartphone: bool = False
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
            patch_size = config.SMARTPHONE_PATCH_SIZE_10x if is_smartphone else config.PATCH_SIZE_10x
            stride = config.SMARTPHONE_PATCH_STRIDE_10x if is_smartphone else config.PATCH_STRIDE_10x
        else:
            patch_size = config.SMARTPHONE_PATCH_SIZE_20x if is_smartphone else config.PATCH_SIZE_20x
            stride = config.SMARTPHONE_PATCH_STRIDE_20x if is_smartphone else config.PATCH_STRIDE_20x
            
        patches = extract_patches_from_image(
            img_path, 
            patch_size=patch_size, 
            stride=stride, 
            is_smartphone=is_smartphone
        )
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
