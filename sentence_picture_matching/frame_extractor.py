import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path

def extract_regions(frame):
    """
    Extract different regions from the frame:
    - text region (top)
    - left image region
    - right image region
    Assumes consistent layout across all frames
    """
    height, width = frame.shape[:2]
    
    # Define regions (adjust these based on your exact layout)
    text_region = frame[0:height//3, :]  # Top third for text
    
    # Remove timer region (top left corner)
    text_region[0:height//10, 0:width//10] = 255  # Make timer region white
    
    return {
        'text': text_region,
    }

def extract_frames(video_path, timestamps_file, output_dir):
    """
    Extract frames from video based on start and end timestamps.
    Save text and image regions separately.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read timestamps file
    df = pd.read_excel(timestamps_file)
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Create directory for this item
    images_dir = os.path.join(output_dir, "frames")
    Path(images_dir).mkdir(exist_ok=True)

    for _, row in df.iterrows():
        item_name = row['Item_name']
        start_time = float(row['Start_time'])
        
        # Get frame at start time (plus small offset to ensure content is visible)
        frame_num = int((start_time + 0.5) * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        
        if ret:
            # Extract regions
            regions = extract_regions(frame)
            
            # Save each region separately
            cv2.imwrite(os.path.join(images_dir, f"{item_name}_text.jpg"), regions['text'])
    
    video.release()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frame regions from video based on timestamps')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('timestamps_file', help='Path to the Excel file containing timestamps')
    parser.add_argument('output_dir', help='Directory to save extracted frames')
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.timestamps_file, args.output_dir)

if __name__ == "__main__":
    main()

'''
FRENCH ballon
python3 frame_extractor.py \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_ballon/screen_videos/ground_thruth/Example_G10116_SPM_FR_video.mkv \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_ballon/screen_videos/ground_thruth/Time_slots_G10116.xlsx \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_ballon/screen_videos/output_annotation/frames

FRENCH balle
python3 frame_extractor.py \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/ground_thruth/G10123-scrrec.mkv \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/ground_thruth/Time_slots_G10123.xlsx \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/output_annotation/frames 

RUSSIAN exceptions
python3 frame_extractor.py \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/ru_exceptional/screen_videos/ground_truth/G30706RU-scrrec.mkv \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/ru_exceptional/screen_videos/ground_truth/Time_slots_G30706RU.xlsx \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/ru_exceptional/screen_videos/output_annotation 
'''