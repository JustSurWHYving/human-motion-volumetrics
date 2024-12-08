import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO, solutions

class HumanTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model = YOLO('models/yolo11n.pt')
        self.heatmap = None
        self.frame_count = 0
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        if not ret:
            return
            
        self.heatmap = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO inference
            results = self.model(frame)
            
            # Process detections
            human_count = 0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Filter for humans (class 0 in COCO dataset)
                    if box.cls == 0:
                        human_count += 1
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Update heatmap
                        self.heatmap[y1:y2, x1:x2] += 1
            
            # Display human count
            cv2.putText(frame, f'Humans: {human_count}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Human Tracking', frame)
            self.frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def generate_heatmap(self):
        # Normalize heatmap
        if self.frame_count > 0:
            self.heatmap = self.heatmap / self.frame_count
            
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            cv2.COLORMAP_JET
        )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        plt.title('Human Detection Heatmap')
        plt.colorbar()
        plt.show()