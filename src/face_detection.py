import cv2
import numpy as np

# Load the Haar Cascade Face Detector from OpenCV's built-in data directory
try:
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("Failed to load frontal face cascade classifier.")
except Exception as e:
    # Fallback to local or default if cv2.data.haarcascades is not configured correctly
    face_cascade = cv2.CascadeClassifier()

def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    Detects faces in an image (numpy array BGR or RGB).
    Returns:
        List of bounding boxes: [(x, y, w, h), ...]
    """
    if image is None or image.size == 0:
        return []
        
    # Convert to grayscale if it is a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Run detector
    # minSize controls filtering small objects (like background elements)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    
    # Return as list of standard python tuples
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def crop_face(image, bbox, margin_percent=0.15):
    """
    Crops the face from the image given a bounding box, with an optional safety margin.
    Args:
        image: Source numpy array image
        bbox: (x, y, w, h) tuple
        margin_percent: Percentage of face dimensions to expand the crop box (for context)
    Returns:
        cropped_face: Cropped face numpy array
    """
    if image is None or image.size == 0 or bbox is None:
        return None
        
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    
    # Calculate margins
    mx = int(w * margin_percent)
    my = int(h * margin_percent)
    
    # Apply margin and clip to image boundaries
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    return cropped
