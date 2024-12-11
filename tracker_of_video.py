import cv2
import torch
from ultralytics import YOLO
import numpy as np

class HumanTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CFG = {"model": "models/yolo11n.pt"}
        self.model = YOLO(self.CFG["model"])
        self.frame_count = 0
        self.prev_frame = None
        self.heatmap = np.zeros((int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        ret, first_frame = cap.read()
        if not ret:
            return
        
        self.prev_frame = cv2.GaussianBlur(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate optical flow with noise reduction
            current_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            flow = cv2.calcOpticalFlowFarneback(self.prev_frame, current_frame, None, 
                                              pyr_scale=0.5, levels=5, winsize=21, 
                                              iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
            
            # Apply threshold to filter out small movements
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitude[magnitude < 1.0] = 0  # Filter small movements
            
            # Update heatmap based on flow magnitude
            self.heatmap = cv2.add(self.heatmap, magnitude)
            
            # Normalize and colorize heatmap
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_frame = cv2.applyColorMap(normalized_heatmap.astype(np.uint8), cv2.COLORMAP_PARULA)

            # Overlay the heatmap on the original frame
            final_frame = cv2.addWeighted(frame, 0.7, heatmap_frame, 0.5, 0)
            
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
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display human count
            cv2.putText(final_frame, f'Humans: {human_count}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Human Tracking', final_frame)
            self.frame_count += 1
            
            # Update previous frame
            self.prev_frame = current_frame
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()