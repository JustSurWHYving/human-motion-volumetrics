import os
import random
from tracker_of_live import HumanTracker

def main():
    # If running on a video, put the video path in the HumanTracker class
    # video_path = 
    
    tracker = HumanTracker()
    tracker.process_video()

# Direct execution check
if __name__ == "__main__":
    main()
