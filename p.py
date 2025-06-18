# Dari fruit_detector.py (bagian detect_banana_ripeness)
def detect_banana_ripeness(img):
    result = img.copy()
    
    ripeness_level_info = {
        0: {"label": "Level 1 (Hijau)", "color": (0, 255, 0)},
        1: {"label": "Level 2 (Hijau Kekuningan)", "color": (50, 255, 0)},
        2: {"label": "Level 3 (Kuning Kehijauan)", "color": (100, 255, 0)},
        3: {"label": "Level 4 (Kuning Penuh)", "color": (0, 255, 255)},
        4: {"label": "Level 5 (Kuning dengan Bintik Kecil)", "color": (0, 200, 255)},
        5: {"label": "Level 6 (Bintik Cokelat Banyak)", "color": (0, 100, 255)},
        6: {"label": "Level 7 (Sangat Matang/Cokelat)", "color": (0, 0, 255)},
    }

    # Get banana mask from segment_fruits
    banana_masks_from_segmentation = segment_fruits(img)
    general_banana_mask = banana_masks_from_segmentation.get('banana')
    
    potential_banana_contours, _ = cv2.findContours(general_banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_banana_bboxes = []
    for banana_cnt in potential_banana_contours:
        area = cv2.contourArea(banana_cnt)
        if area > 300: # Adjusted threshold for banana detection
            x, y, w, h = cv2.boundingRect(banana_cnt)
            valid_banana_bboxes.append((x, y, w, h))
            
    valid_banana_bboxes.sort(key=lambda bbox: bbox[0]) # Sort by x-coordinate

    for i, bbox in enumerate(valid_banana_bboxes):
        if i >= len(ripeness_level_info): # Stop if more than 7 bananas, or if labels run out
            break
        x, y, w, h = bbox
        ripeness_idx = i # Direct mapping
        
        label_info = ripeness_level_info[ripeness_idx]
        text_label = label_info["label"]
        color = label_info["color"]

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return result