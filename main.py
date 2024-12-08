import os
import random
from tracker import HumanTracker

def main():
    # Set dataset location
    video_folder = "videos/"
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Russian roullete with the dataset videos
    random_video = random.choice(video_files)
    random_video_path = os.path.join(video_folder, random_video)
    
    tracker = HumanTracker(random_video_path)
    tracker.process_video()

# Direct execution check
if __name__ == "__main__":
    main()
