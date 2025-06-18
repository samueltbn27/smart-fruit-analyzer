import cv2
import numpy as np

def preprocess_image(img):
    """Normalize image size and enhance contrast using CLAHE."""
    # Resize only if necessary, or make it dynamic based on original size
    # For now, keep original size from main, if it's already large enough,
    # or make the resize optional.
    # If the image is too large, it might slow down processing.
    # Let's keep a consistent size for processing.
    # It's better to process a fixed size image for consistent results.
    img_resized = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA) # Added interpolation for better quality when shrinking
    
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) # Increased clipLimit slightly for stronger effect
    limg = clahe.apply(l)
    
    # Merge the enhanced L-channel with original A and B channels
    enhanced_lab = cv2.merge([limg, a, b])
    
    # Convert back to BGR
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def segment_fruits(img):
    """Segment fruits by color in HSV space with refined ranges, especially for banana."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = {}

    # --- APPLE RANGES ---
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    lower_green_apple = np.array([60, 80, 60])
    upper_green_apple = np.array([90, 255, 255])
    mask_green_apple = cv2.inRange(hsv, lower_green_apple, upper_green_apple)

    masks['apple'] = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), mask_green_apple)

    # --- ORANGE RANGES ---
    lower_orange = np.array([10, 150, 100])
    upper_orange = np.array([25, 255, 255])
    masks['orange'] = cv2.inRange(hsv, lower_orange, upper_orange)

    # --- BANANA RANGES (Broadened to include green, yellow, and brown tones) ---
    lower_banana_general = np.array([15, 40, 40])
    upper_banana_general = np.array([60, 255, 255])
    masks['banana'] = cv2.inRange(hsv, lower_banana_general, upper_banana_general)

    return masks

def detect_edges(img):
    """Detect edges using Canny algorithm"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)