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
    # left_image = frame[height//3:, 0:width//2]  # Bottom left for left image
    # right_image = frame[height//3:, width//2:]  # Bottom right for right image
    
    # Remove timer region (top left corner)
    text_region[0:height//10, 0:width//10] = 255  # Make timer region white
    
    return {
        'text': text_region,
        # 'left_image': left_image,
        # 'right_image': right_image
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
    items_dir = os.path.join(output_dir, "frames")
    Path(items_dir).mkdir(exist_ok=True)

    for _, row in df.iterrows():
        item_name = row['Item_name']
        start_time = float(row['Start_time'])
        

        
        # Get frame at start time (plus small offset to ensure content is visible)
        frame_num = int((start_time + 0.1) * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        
        if ret:
            # Extract regions
            regions = extract_regions(frame)
            
            # Save each region separately
            cv2.imwrite(os.path.join(items_dir, f"{item_name}_text.jpg"), regions['text'])
            # cv2.imwrite(os.path.join(item_dir, f"{item_name}_left.jpg"), regions['left_image'])
            # cv2.imwrite(os.path.join(item_dir, f"{item_name}_right.jpg"), regions['right_image'])
            
            # Save full frame for reference
            # cv2.imwrite(os.path.join(item_dir, f"{item_name}_full.jpg"), frame)
    
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