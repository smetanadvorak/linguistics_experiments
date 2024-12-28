import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class RegionConfig:
    """Configuration for a region to extract from the frame"""
    top: float
    bottom: float
    left: float
    right: float
    clear_timer: bool = False
    timer_height: Optional[int] = None
    timer_width: Optional[int] = None

class FrameExtractor:
    def __init__(self, regions: Dict[str, RegionConfig]):
        """
        Initialize the FrameExtractor with region configurations.

        Args:
            regions: Dictionary mapping region names to their configurations
                    e.g., {'text': RegionConfig(top=0, bottom=100, left=0, right=800)}
        """
        self.regions = regions

    def extract_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract different regions from the frame based on configuration.

        Args:
            frame: Input frame as numpy array

        Returns:
            Dictionary mapping region names to extracted numpy arrays
        """
        height, width = frame.shape[:2]
        extracted = {}

        for name, config in self.regions.items():
            # Convert relative positions to absolute if necessary
            top = int(height * config.top)
            bottom = int(height * config.bottom)
            left = int(width * config.left)
            right = int(width * config.right)

            # Extract region
            region = frame[top:bottom, left:right].copy()

            # Clear timer area if specified
            if config.clear_timer:
                timer_h = config.timer_height if config.timer_height else height // 10
                timer_w = config.timer_width if config.timer_width else width // 10
                region[0:timer_h, 0:timer_w] = 255

            extracted[name] = region

        return extracted

    def extract_frames(self, video_path: str, timestamps_file: str, output_dir: str,
                      time_offset: float = 0.5) -> None:
        """
        Extract frames from video based on timestamps and save regions.

        Args:
            video_path: Path to the video file
            timestamps_file: Path to the Excel file containing timestamps
            output_dir: Directory to save extracted frames
            time_offset: Time offset in seconds to add to start time (default: 0.5)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Read timestamps file
        df = pd.read_excel(timestamps_file)

        # Open video file
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        # Create directory for frames
        images_dir = os.path.join(output_dir, "frames")
        Path(images_dir).mkdir(exist_ok=True)

        for _, row in df.iterrows():
            item_name = row['Item_name']
            start_time = float(row['Start_time'])

            # Get frame at start time plus offset
            frame_num = int((start_time + time_offset) * fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video.read()

            if ret:
                # Extract regions
                regions = self.extract_regions(frame)

                # Save each region
                for region_name, region_img in regions.items():
                    output_path = os.path.join(images_dir, f"{item_name}.jpg")
                    cv2.imwrite(output_path, region_img)

        video.release()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract frame regions from video based on timestamps')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('timestamps_file', help='Path to the Excel file containing timestamps')
    parser.add_argument('output_dir', help='Directory to save extracted frames')

    args = parser.parse_args()

    # Example configuration
    regions = {
        'text': RegionConfig(
            top=0,
            bottom=1/3,  # One third of height
            left=0,
            right=1,     # Full width
            clear_timer=True
        )
    }

    extractor = FrameExtractor(regions)
    extractor.extract_frames(args.video_path, args.timestamps_file, args.output_dir)

if __name__ == "__main__":
    main()