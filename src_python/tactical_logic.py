"""
tactical_logic.py
=================
Tactical Decision Logic & Spatial Non-Maximum Suppression (NMS).

Applies spatial filtering and NMS thresholding over C++ anomaly heatmaps to discard
transient false positives (falling debris, atmospheric smoke bursts, camera shake)
and isolate high-confidence battlefield threat detections.
"""

import cv2
import numpy as np


class TacticalDecisionLogic:
    """
    Filters spatial MSE anomaly heatmaps, runs contour clustering, and applies NMS.
    """
    def __init__(self, mse_threshold=0.06, min_area=150, max_area=15000, nms_iou_threshold=0.3):
        self.mse_threshold = mse_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.nms_iou_threshold = nms_iou_threshold

    def set_mse_threshold(self, th):
        self.mse_threshold = th

    def compute_iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def apply_nms(self, boxes, scores):
        """Performs Non-Maximum Suppression (NMS) over bounding boxes."""
        if len(boxes) == 0:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)

        # Sort boxes by anomaly score descending
        idxs = np.argsort(scores_arr)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)

            if len(idxs) == 1:
                break

            rest = idxs[1:]
            filtered_idxs = []
            for i in rest:
                iou = self.compute_iou(boxes_arr[current], boxes_arr[i])
                if iou < self.nms_iou_threshold:
                    filtered_idxs.append(i)
            idxs = np.array(filtered_idxs)

        keep_boxes = [boxes[i] for i in keep]
        keep_scores = [scores[i] for i in keep]
        return keep_boxes, keep_scores

    def detect_tactical_threats(self, mse_heatmap):
        """
        Processes C++ MSE anomaly matrix and returns consolidated threat alerts.
        
        Args:
            mse_heatmap (np.ndarray): Pixel-wise MSE matrix (float32, 0.0 - 1.0)
            
        Returns:
            threats (list): List of dicts [{'bbox': [x, y, w, h], 'score': float, 'type': str}]
            binary_mask (np.ndarray): Binary thresholded anomaly mask
        """
        if mse_heatmap is None or mse_heatmap.size == 0:
            return [], None

        # 1. Adaptive / Absolute Thresholding
        binary_mask = (mse_heatmap > self.mse_threshold).astype(np.uint8) * 255

        # 2. Morphological filtering to discard transient visual noise (smoke/dust bursts)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour Detection & Candidate Extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_boxes = []
        candidate_scores = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Compute localized mean anomaly intensity score
                roi_mse = mse_heatmap[y:y+h, x:x+w]
                score = float(np.mean(roi_mse)) if roi_mse.size > 0 else float(self.mse_threshold)
                
                candidate_boxes.append([x, y, w, h])
                candidate_scores.append(score)

        # 4. Apply Spatial Non-Maximum Suppression (NMS)
        nms_boxes, nms_scores = self.apply_nms(candidate_boxes, candidate_scores)

        # 5. Format Tactical Threat Alerts
        threats = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x, y, w, h = box
            
            # Threat Classification based on area and anomaly score
            area = w * h
            if area > 1200:
                classification = "HOSTILE ARMORED ASSET / VEHICLE"
            elif area > 400:
                classification = "CONCEALED INFANTRY / PERSONNEL"
            else:
                classification = "DISTURBED GROUND / IMMINENT HAZARD"
                
            threats.append({
                'id': i + 1,
                'bbox': [x, y, w, h],
                'score': score,
                'type': classification,
                'confidence': min(1.0, score / (self.mse_threshold * 2.5))
            })

        return threats, cleaned_mask
