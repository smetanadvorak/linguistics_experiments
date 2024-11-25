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
        reference_frames[event_file.name] = {
            'text': cv2.imread(str(event_file))
        }
    return reference_frames

def compute_text_similarity(text1, text2):
    """Compute text similarity using OCR-friendly comparison"""
    # Convert to grayscale and threshold to binary
    gray1 = cv2.cvtColor(text1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(text2, cv2.COLOR_BGR2GRAY)
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

def extract_regions(frame):
    """Extract text region from frame"""
    height = frame.shape[0]
    text_region = frame[0:height//3, :].copy()
    text_region[0:height//10, 0:frame.shape[1]//10] = 255  # Remove timer region
    return {'text': text_region}

def find_events(video_path, reference_frames, threshold=0.8, text_weight=1.0):
    """Find events in video using text similarity"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_events = defaultdict(lambda: {'start': None, 'score': 0})
    detected_events = []
    
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
            
            for event_name, ref_data in reference_frames.items():
                text_sim = compute_text_similarity(regions['text'], ref_data['text'])
                if text_sim > threshold:
                    if current_events[event_name]['start'] is None:
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
    
    video.release()
    detected_events.sort(key=lambda x: x['start_time'])
    return detected_events

def process_video(video_path, reference_frames, output_path, threshold, text_weight):
    """Process a single video file"""
    video_name = Path(video_path).stem
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
    
    skip = [
        "G10119-scrrec.mkv",
        "G10123-scrrec.mkv",
        "G10114-scrrec.mkv",
        "G20418FR-scrrec.mkv",
        "G10108-scrrec.mkv",
        "G10126-scrrec.mkv",
        "G20419FR-scrrec.mkv",
        "G30714FR-scrrec.mkv",
        "G20413FR-scrrec.mkv",
        "G20414FR-scrrec.mkv",
        "G30713FR-scrrec.mkv",
        "G10107-scrrec.mkv",
        "G30706FR-scrrec.mkv",
        "G20406FR-scrrec.mkv",
        "G10116-scrrec.mkv",
        "G10121-scrrec.mkv",
        "G20425FR-scrrec.mkv",
        "G20407FR-scrrec.mkv",
        "G30707FR-scrrec.mkv",
        "G30712FR-scrrec.mkv",
        "G20415FR-scrrec.mkv",
        "G20412FR-scrrec.mkv",
        "G30715FR-scrrec.mkv",
        "G10124-scrrec.mkv",
        "G10113-scrrec.mkv"
    ]

    args = parser.parse_args()
    reference_frames = load_reference_frames(args.reference_dir)
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        output_file = process_video(
            input_path, reference_frames,
            args.output_path, args.threshold, args.text_weight
        )
        print(f"Saved events to: {output_file}")
    
    elif input_path.is_dir():
        video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi')) + \
                     list(input_path.glob('*.mkv')) + list(input_path.glob('*.mov'))
        for video_file in tqdm(video_files, desc="Processing videos"):
            if video_file.name in skip:
                continue
            output_file = process_video(
                video_file, reference_frames,
                args.output_path, args.threshold, args.text_weight
            )
            
    else:
        print(f"Error: '{input_path}' is not a valid file or directory")
        exit(1)

if __name__ == "__main__":
    main()