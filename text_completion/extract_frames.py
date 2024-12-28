from eyeoi.frame_extractor import FrameExtractor, RegionConfig

import argparse

parser = argparse.ArgumentParser(description='Extract frame regions from video based on timestamps')
parser.add_argument('video_path', help='Path to the video file')
parser.add_argument('timestamps_file', help='Path to the Excel file containing timestamps')
parser.add_argument('output_dir', help='Directory to save extracted frames')

args = parser.parse_args()

# Example configuration
regions = {
    'text': RegionConfig(
        top=0.0, bottom=1/3, left=0, right=1,
        clear_timer=True
    )
}

extractor = FrameExtractor(regions)
extractor.extract_frames(args.video_path, args.timestamps_file, args.output_dir)
