import cv2
from ultralytics import YOLO, solutions

class HumanTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.CFG = {"model": "models/yolo11n.pt"}
        self.model = YOLO(self.CFG["model"])
        self.frame_count = 0
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        ret, first_frame = cap.read()
        if not ret:
            return
        
        # Initialize heatmap
        self.heatmap = solutions.Heatmap(
            show=False,
            model= self.CFG["model"],
            colormap=cv2.COLORMAP_PARULA,
        )
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Generate heatmap
            heatmap_frame = self.heatmap.generate_heatmap(frame)

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
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()