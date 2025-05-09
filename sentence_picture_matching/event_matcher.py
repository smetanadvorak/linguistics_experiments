import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm

def load_reference_frames(reference_dir):
    """Load reference frames from directory"""
    reference_frames = {}
    for event_file in Path(reference_dir).iterdir():
        if event_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:  # Only process image files
            img = cv2.imread(str(event_file))
            if img is not None:  # Check if image was loaded successfully
                reference_frames[event_file.name] = {
                    'text': img
                }
            else:
                print(f"Warning: Failed to load reference image: {event_file}")
    
    if not reference_frames:
        raise ValueError(f"No valid reference frames found in {reference_dir}")
    
    return reference_frames

def compute_text_similarity(text1, text2):
    """Compute text similarity using OCR-friendly comparison"""
    # Validate inputs
    if text1 is None or text2 is None:
        return 0.0
    
    try:
        # Convert to grayscale and threshold to binary
        gray1 = cv2.cvtColor(text1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(text2, cv2.COLOR_BGR2GRAY)
        
        # Ensure images are the same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        _, binary1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)
        _, binary2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Check for empty frames
        text_pixels1 = np.count_nonzero(binary1) / binary1.size
        text_pixels2 = np.count_nonzero(binary2) / binary2.size
        if text_pixels1 < 0.001 or text_pixels2 < 0.001:
            return 0.0
        
        # Compare using normalized cross-correlation
        result = cv2.matchTemplate(binary1, binary2, cv2.TM_CCORR_NORMED)
        return float(result.max())
    except Exception as e:
        print(f"Warning: Error computing similarity: {str(e)}")
        return 0.0

def extract_regions(frame):
    """Extract text region from frame"""
    if frame is None:
        return {'text': None}
    
    try:
        height = frame.shape[0]
        text_region = frame[0:height//3, :].copy()
        text_region[0:height//10, 0:frame.shape[1]//10] = 255  # Remove timer region
        return {'text': text_region}
    except Exception as e:
        print(f"Warning: Error extracting regions: {str(e)}")
        return {'text': None}

def find_events(video_path, reference_frames, threshold=0.8, text_weight=1.0):
    """Find events in video using text similarity"""
    if not reference_frames:
        raise ValueError("No reference frames provided")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_events = defaultdict(lambda: {'start': None, 'score': 0})
    detected_events = []

    ref_frames_copy = dict(reference_frames)
    
    with tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}") as pbar:
        frame_number = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            current_time = frame_number / fps
            frame_number += 1
            pbar.update(1)
            
            regions = extract_regions(frame)
            if regions['text'] is None:
                continue
        
            to_delete = None
            for event_name, ref_data in ref_frames_copy.items():
                text_sim = compute_text_similarity(regions['text'], ref_data['text'])
                if text_sim > threshold:
                    if current_events[event_name]['start'] is None:
                        print("Detected", event_name)
                        current_events[event_name]['start'] = current_time
                    current_events[event_name]['score'] += text_sim
                else:
                    if current_events[event_name]['start'] is not None:
                        detected_events.append({
                            'event': event_name,
                            'start_time': float(current_events[event_name]['start']),
                            'end_time': float(current_time),
                            'confidence': float(current_events[event_name]['score'] / 
                                        ((current_time - current_events[event_name]['start']) * fps))
                        })
                        current_events[event_name] = {'start': None, 'score': 0}
                        to_delete = event_name
            
            if to_delete is not None:
                del ref_frames_copy[to_delete]  # for a little bit of speed 

    video.release()
    detected_events.sort(key=lambda x: x['start_time'])
    return detected_events

def process_video(video_path, reference_frames, output_path, threshold, text_weight):
    """Process a single video file"""
    try:
        video_name = Path(video_path).stem
        video_name = video_name.replace("Copy of ", "")
        output_file = Path(output_path) / f"{video_name}_events.json"
        if video_name.endswith('_events'):
            output_file = Path(output_path) / f"{video_name}.json"

        detected_events = find_events(
            str(video_path),
            reference_frames,
            threshold=threshold,
            text_weight=text_weight
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(detected_events, f, indent=2)
        
        return output_file
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find events in videos')
    parser.add_argument('input_path', help='Video file or folder of videos')
    parser.add_argument('reference_dir', help='Directory with reference frames')
    parser.add_argument('output_path', help='Path for detected events JSON')
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Similarity threshold (default: 0.8)')
    parser.add_argument('--text-weight', type=float, default=1.0,
                      help='Text similarity weight (default: 1.0)')
    
    args = parser.parse_args()
    
    try:
        reference_frames = load_reference_frames(args.reference_dir)
        input_path = Path(args.input_path)
        
        if input_path.is_file():
            output_file = process_video(
                input_path, reference_frames,
                args.output_path, args.threshold, args.text_weight
            )
            if output_file:
                print(f"Saved events to: {output_file}")
        
        elif input_path.is_dir():
            video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi')) + \
                         list(input_path.glob('*.mkv')) + list(input_path.glob('*.mov'))
            
            if not video_files:
                print(f"No video files found in {input_path}")
                return
            
            for video_file in tqdm(video_files, desc="Processing videos"):
                output_file = process_video(
                    video_file, reference_frames,
                    args.output_path, args.threshold, args.text_weight
                )
                
        else:
            print(f"Error: '{input_path}' is not a valid file or directory")
            exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()


'''
FRENCH ballon

FRENCH balle
python3 event_matcher.py \ 
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/input_videos \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/output_annotation/frames \
    /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr_balle/screen_videos/output_annotation \
    --threshold 0.95


python3 event_matcher.py \ 
    /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/screen_videos \
    /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/output_annotation/frames \
    /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/events \
    --threshold 0.95

python3 event_matcher.py /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/screen_videos /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/output_annotation/frames /Users/akmbpro/Documents/coding/alina/sentence_picture_matching/data/aoi2/en/events --threshold 0.95

python3 event_matcher.py /Users/akmbpro/Documents/coding/alina/log_parsing/data2/videos_en /Users/akmbpro/Documents/coding/alina/log_parsing/data2/frames_en /Users/akmbpro/Documents/coding/alina/log_parsing/data2/en/events --threshold 0.95

'''