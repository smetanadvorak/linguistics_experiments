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
        top=0.45, bottom=0.55, 
        left=0.45, right=0.55,
        clear_timer=False
    )
}

regions_250405 = {
    'text': RegionConfig(
        top=0.25, bottom=0.75, 
        left=0.3, right=0.7,
        clear_timer=False
    )
}

extractor = FrameExtractor(regions_250405)
extractor.extract_frames(args.video_path, args.timestamps_file, args.output_dir)


"""
python3 extract_frames.py \
    data_250405/en/G10207EN.mkv \
    data_250405/en/Time_slots_G10207_EndPoint.xlsx \
    data_250405/en/events
"""