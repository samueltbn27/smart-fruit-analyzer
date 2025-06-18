import cv2
import numpy as np
from image_processor import segment_fruits

def color_based_detection(img, target_fruit="All Fruits"):
    """Detect fruits based on color ranges and draw bounding boxes.
       Handles potential overlaps by processing masks in a specific order or by area.
    """
    masks = segment_fruits(img)
    result = img.copy()
    
    detected_fruits_info = []

    for fruit, mask in masks.items():
        if target_fruit == "All Fruits" or target_fruit.lower() == fruit:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000: # Filter small noise (adjust threshold as needed)
                    x, y, w, h = cv2.boundingRect(cnt)
                    detected_fruits_info.append({
                        'label': fruit.capitalize(),
                        'bbox': (x, y, w, h),
                        'mask_area': area,
                        'fruit_type': fruit 
                    })
    
    for fruit_info in detected_fruits_info:
        label = fruit_info['label']
        x, y, w, h = fruit_info['bbox']
        fruit_type_raw = fruit_info['fruit_type']

        box_color = (0, 255, 0) # Default green

        if fruit_type_raw == "apple":
            box_color = (0, 0, 255) # Red for apple (B,G,R)
        elif fruit_type_raw == "orange":
            box_color = (0, 165, 255) # Orange for orange (B,G,R)

        cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(result, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    return result

def detect_banana_ripeness(img):
    """Detect banana ripeness by sorting detected bananas from left to right."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = img.copy()

    # Define labels and colors for 7 ripeness levels
    ripeness_level_info = [
        {"label": "Level 1 (Hard Green)", "box_color": (0, 255, 0)},
        {"label": "Level 2 (Green-Yellowish)", "box_color": (0, 200, 50)},
        {"label": "Level 3 (Yellow-Green)", "box_color": (0, 165, 255)},
        {"label": "Level 4 (Fully Yellow)", "box_color": (0, 255, 255)},
        {"label": "Level 5 (Yellow w/ Few Spots)", "box_color": (0, 255, 200)},
        {"label": "Level 6 (Yellow w/ Many Spots)", "box_color": (0, 180, 255)},
        {"label": "Level 7 (Mostly Spotted)", "box_color": (50, 100, 255)}
    ]

    banana_masks_from_segmentation = segment_fruits(img)
    general_banana_mask = banana_masks_from_segmentation.get('banana')

    if general_banana_mask is None:
        print("Warning: No banana mask from segmentation. Using broad fallback range.")
        # Fallback range is already broad in segment_fruits, so this might not be needed much.
        # But keeping it for robustness.
        general_banana_lower = np.array([15, 40, 40])
        general_banana_upper = np.array([60, 255, 255])
        general_banana_mask = cv2.inRange(hsv, general_banana_lower, general_banana_upper)

    potential_banana_contours, _ = cv2.findContours(general_banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_banana_bboxes = []
    # Lower the area threshold to catch smaller or less-defined contours
    for banana_cnt in potential_banana_contours:
        area = cv2.contourArea(banana_cnt)
        if area > 300: # Changed from 500 to 300 - adjust if needed
            x, y, w, h = cv2.boundingRect(banana_cnt)
            valid_banana_bboxes.append((x, y, w, h))
    
    # Sort bananas by their X-coordinate (left to right)
    valid_banana_bboxes.sort(key=lambda bbox: bbox[0]) 

    # Assign ripeness levels based on sorted order
    num_bananas = len(valid_banana_bboxes)
    
    # Only process up to 7 bananas, or the number of bananas found
    # This loop directly maps sorted index to ripeness level.
    for i, bbox in enumerate(valid_banana_bboxes):
        if i >= len(ripeness_level_info): # Stop if we run out of defined levels
            break

        x, y, w, h = bbox
        
        # Direct mapping: banana at index 0 gets Level 1, index 1 gets Level 2, etc.
        ripeness_idx = i 
        
        ripeness_label = ripeness_level_info[ripeness_idx]["label"]
        box_color = ripeness_level_info[ripeness_idx]["box_color"]
        
        # Draw the bounding box and label
        cv2.rectangle(result, (x,y), (x+w,y+h), box_color, 2)
        cv2.putText(result, ripeness_label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    return result