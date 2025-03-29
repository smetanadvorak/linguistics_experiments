import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from matplotlib import pyplot as plt

def load_reference_frames(reference_dir):
    """Load reference frames from directory"""
    reference_frames = {}
    for event_file in Path(reference_dir).iterdir():
        if event_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = cv2.imread(str(event_file))
            if img is not None:
                # Pre-process reference frames once to avoid repeated processing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # Downsample reference images to speed up matching
                downsample_factor = 0.5
                small_binary = cv2.resize(binary, (0, 0), fx=downsample_factor, fy=downsample_factor)
                
                # Calculate non-zero pixels ratio for empty frame detection
                text_pixels = np.count_nonzero(small_binary) / small_binary.size
                
                reference_frames[event_file.name] = {
                    'binary': small_binary,
                    'text_pixels': text_pixels
                }
            else:
                print(f"Warning: Failed to load reference image: {event_file}")
    
    if not reference_frames:
        raise ValueError(f"No valid reference frames found in {reference_dir}")
    
    return reference_frames

def compute_text_similarity(frame_binary, frame_pixels, ref_binary, ref_pixels):
    """Compute text similarity using faster comparison"""
    # Check for empty frames early to avoid unnecessary computation
    if frame_pixels < 0.003: #or ref_pixels < 0.001:
        return 0.0
    
    try:
        # Ensure images are the same size
        if frame_binary.shape != ref_binary.shape:
            ref_binary = cv2.resize(ref_binary, (frame_binary.shape[1], frame_binary.shape[0]))
        
        # Use a faster correlation method (TM_CCOEFF_NORMED is slightly faster than TM_CCORR_NORMED)
        # For text matching, SQDIFF_NORMED might also work well and is faster
        result = cv2.matchTemplate(frame_binary, ref_binary, cv2.TM_CCOEFF_NORMED)
        return float(result.max())
    except Exception as e:
        print(f"Warning: Error computing similarity: {str(e)}")
        return 0.0

def extract_regions(frame, downsample_factor=0.5):
    """Extract text region from frame with downsampling"""
    if frame is None:
        return {'binary': None, 'pixels_ratio': 0}
    
    try:
        height = frame.shape[0]
        text_region = frame[0:height//3, :].copy()
        text_region[0:height//10, 0:frame.shape[1]//10] = 255  # Remove timer region
        
        # Pre-process for faster matching
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Downsample to speed up matching
        small_binary = cv2.resize(binary, (0, 0), fx=downsample_factor, fy=downsample_factor)

        # Pre-calculate pixel ratio
        pixels_ratio = np.count_nonzero(small_binary) / small_binary.size
        # print(pixels_ratio)
                
        # plt.figure()
        # plt.imshow(small_binary)
        # plt.show()
        
        return {'binary': small_binary, 'pixels_ratio': pixels_ratio}
    except Exception as e:
        print(f"Warning: Error extracting regions: {str(e)}")
        return {'binary': None, 'pixels_ratio': 0}

def process_frame_batch(frame_data, reference_frames, threshold, current_events, batch_results):
    """Process a batch of frames to detect events"""
    for frame_info in frame_data:
        frame_number, current_time, regions = frame_info
        
        if regions['binary'] is None:
            continue
        
        for event_name, ref_data in reference_frames.items():
            if event_name in batch_results['to_delete']:
                continue
                
            text_sim = compute_text_similarity(
                regions['binary'], 
                regions['pixels_ratio'],
                ref_data['binary'], 
                ref_data['text_pixels']
            )
            
            if text_sim > threshold:
                if event_name not in current_events or current_events[event_name]['start'] is None:
                    batch_results['events_to_start'].append((event_name, current_time))
                batch_results['scores'][event_name] = batch_results['scores'].get(event_name, 0) + text_sim
            else:
                if event_name in current_events and current_events[event_name]['start'] is not None:
                    batch_results['events_to_end'].append({
                        'event': event_name,
                        'start_time': float(current_events[event_name]['start']),
                        'end_time': float(current_time),
                        'confidence': float(current_events[event_name]['score'] / 
                                    ((current_time - current_events[event_name]['start']) * batch_results['fps']))
                    })
                    batch_results['to_delete'].add(event_name)

def find_events_parallel(video_path, reference_frames, threshold=0.8, batch_size=10, num_workers=None):
    """Find events in video using parallel processing for batches of frames"""
    if not reference_frames:
        raise ValueError("No reference frames provided")
    
    # Determine optimal number of workers if not specified
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU for the OS
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_events = {}
    detected_events = []
    
    # Set up parallel processing pool
    pool = mp.Pool(processes=num_workers)
    
    with tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}") as pbar:
        frame_number = 0
        
        while True:
            # Process frames in batches
            frame_batch = []
            for _ in range(batch_size):
                ret, frame = video.read()
                if not ret:
                    break
                    
                current_time = frame_number / fps
                frame_number += 1
                pbar.update(1)
                
                regions = extract_regions(frame)
                frame_batch.append((frame_number, current_time, regions))
            
            if not frame_batch:
                break
            
            # Prepare shared data structure for batch processing results
            batch_results = {
                'events_to_start': [],
                'events_to_end': [],
                'scores': {},
                'to_delete': set(),
                'fps': fps
            }
            
            # Process frame batch
            process_frame_batch(frame_batch, 
                               {k: v for k, v in reference_frames.items() if k not in batch_results['to_delete']}, 
                               threshold, current_events, batch_results)
            
            # Update current events with batch results
            for event_name, start_time in batch_results['events_to_start']:
                if event_name not in current_events:
                    current_events[event_name] = {'start': None, 'score': 0}
                if current_events[event_name]['start'] is None:
                    print("Detected", event_name)
                    current_events[event_name]['start'] = start_time
                
            for event_name, score in batch_results['scores'].items():
                if event_name in current_events:
                    current_events[event_name]['score'] += score
            
            # Add detected events
            detected_events.extend(batch_results['events_to_end'])
            
            # Remove processed reference frames
            for event_name in batch_results['to_delete']:
                if event_name in current_events:
                    current_events[event_name] = {'start': None, 'score': 0}
                
    video.release()
    pool.close()
    pool.join()
    
    detected_events.sort(key=lambda x: x['start_time'])
    return detected_events

def process_video(video_path, reference_frames, output_path, threshold, num_workers=None):
    """Process a single video file"""
    try:
        video_name = Path(video_path).stem
        video_name = video_name.replace("Copy of ", "")
        output_file = Path(output_path) / f"{video_name}_events.json"
        if video_name.endswith('_events'):
            output_file = Path(output_path) / f"{video_name}.json"

        detected_events = find_events_parallel(
            str(video_path),
            reference_frames,
            threshold=threshold,
            num_workers=num_workers
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
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of frames to process in each batch (default: 10)')
    
    args = parser.parse_args()
    
    try:
        reference_frames = load_reference_frames(args.reference_dir)
        input_path = Path(args.input_path)
        
        if input_path.is_file():
            output_file = process_video(
                input_path, reference_frames,
                args.output_path, args.threshold, args.workers
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
                print(video_file.name)
                if video_file.name[0] == '_':
                    continue
                output_file = process_video(
                    video_file, reference_frames,
                    args.output_path, args.threshold, args.workers
                )
                
        else:
            print(f"Error: '{input_path}' is not a valid file or directory")
            exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()