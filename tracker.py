import cv2
import numpy as np
from collections import defaultdict
from scipy.stats import zscore
import torch
from ultralytics import YOLO

class VolumetricTracker:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.track_history = defaultdict(lambda: [])
        self.mask = None
        
        # Volume tracking parameters
        self.volume_threshold = 500
        self.motion_history = np.zeros((int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))))
        self.volume_data = defaultdict(list)
        
        # Initialize background subtractor with smaller block size
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=8,  # Reduced threshold for finer detection
            detectShadows=False
        )
        
        # Initialize YOLO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('models/yolo11n.pt')
        self.model.conf = 0.3  # confidence threshold
        self.model.classes = [0]  # only detect persons

    def detect_humans(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model(frame_rgb)
        
        # Get bounding boxes
        boxes = []
        
        for detection in results[0].boxes.data:
            if int(detection[5]) == 0:  # class 0 is person
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = float(detection[4])
                boxes.append((x1, y1, x2, y2, confidence))
                
        return boxes

    def update_motion_history(self, boxes, frame):
        # Create mask from human detections with finer granularity
        motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2, _ = box
            # Create a grid of smaller rectangles within the bounding box
            cell_size = 8  # Smaller cell size for finer granularity
            for y in range(y1, y2, cell_size):
                for x in range(x1, x2, cell_size):
                    end_y = min(y + cell_size, y2)
                    end_x = min(x + cell_size, x2)
                    motion_mask[y:end_y, x:end_x] = 1
        
        # Update motion history with faster decay for finer temporal resolution
        self.motion_history = np.maximum(
            self.motion_history * 0.95,  # Faster decay
            motion_mask.astype(float)
        )
        
        # Calculate volumetric regions with smaller minimum area
        ret, thresh = cv2.threshold(
            (self.motion_history * 255).astype(np.uint8),
            20, 255, cv2.THRESH_BINARY  # Lower threshold for finer detection
        )        
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE  # Use APPROX_NONE for finer contours
        )
        
        return contours

    def detect_volume_anomalies(self, volume, region_id):
        self.volume_data[region_id].append(volume)
        if len(self.volume_data[region_id]) > 30:
            volumes = np.array(self.volume_data[region_id][-30:])
            z_scores = zscore(volumes)
            return np.any(np.abs(z_scores) > 3)
        return False

    def track(self):
        prev_time = cv2.getTickCount()
        while self.video.isOpened():
            success, frame = self.video.read()
            if not success:
                break

            # Detect humans using YOLO
            boxes = self.detect_humans(frame)
            
            contours = self.update_motion_history(boxes, frame)
            
            # Apply background subtraction with finer morphological operations
            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8))

            # Visualize motion history
            motion_display = cv2.normalize(
                self.motion_history, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            motion_colored = cv2.applyColorMap(motion_display, cv2.COLORMAP_JET)
            
            # Combine original frame with motion visualization
            alpha = 0.7
            output = cv2.addWeighted(frame, alpha, motion_colored, 1-alpha, 0)

            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output, f'Person: {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(output, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Volumetric Motion Tracking', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()