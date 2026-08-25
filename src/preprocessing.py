import cv2
import numpy as np

def resize_image(image, target_size=(160, 160)):
    """Resizes an image to the target dimensions."""
    if image is None or image.size == 0:
        return None
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image, convert_to_rgb=True):
    """
    Normalizes pixel values of an image.
    Converts image to float32 and scales pixel values to range [-1, 1].
    Converts BGR to RGB by default for deep learning backbone compatibility.
    """
    if image is None or image.size == 0:
        return None
        
    if convert_to_rgb and len(image.shape) == 3:
        # Convert BGR (OpenCV default) to RGB (deep learning default)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
        
    # Convert RGB to float and scale to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    # Center to [-1, 1] which is typical for MobileNet/Inception
    img_normalized = (img_float - 0.5) * 2.0
    return img_normalized

def calculate_blurriness(image):
    """
    Calculates blurriness using the Laplacian variance method.
    Higher values mean more focus (less blurry).
    """
    if image is None or image.size == 0:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Variance of the Laplacian highlights edge sharpness
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_face_quality(image, bbox, min_size=50, blur_threshold=100.0):
    """
    Checks if a detected face meets size and sharpness quality requirements.
    Args:
        image: Original frame/image
        bbox: (x, y, w, h) of the face
        min_size: Minimum width/height in pixels
        blur_threshold: Variance of Laplacian below which image is marked blurry
    Returns:
        is_valid: bool
        reason: str description if invalid, otherwise "OK"
    """
    if bbox is None:
        return False, "No face bounding box provided."
        
    x, y, w, h = bbox
    
    # 1. Size Check
    if w < min_size or h < min_size:
        return False, f"Face too small ({w}x{h}px). Minimum required size is {min_size}x{min_size}px."
        
    # Crop the face to check blurriness
    face_crop = image[y:y+h, x:x+w]
    if face_crop.size == 0:
        return False, "Invalid bounding box boundaries."
        
    # 2. Sharpness/Blurriness Check
    blur_score = calculate_blurriness(face_crop)
    if blur_score < blur_threshold:
        return False, f"Image too blurry (Sharpness score: {blur_score:.1f}). Please stay still."
        
    return True, "OK"
