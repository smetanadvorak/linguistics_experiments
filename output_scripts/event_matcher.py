"""
example:
frame_extraction % python event_matcher.py \
    /Users/akmbpro/Documents/coding/alina/aoi_reorder/frame_extraction/G20407EN-scrrec.mkv \
    reference_frames/ \
    detected_events.json \
    --threshold 0.88 \
    --text-weight 1.0
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from collections import defaultdict

def load_reference_frames(reference_dir):
    """
    Load reference frames for each event, separated by region
    """
    reference_frames = {}
    
    for event_file in Path(reference_dir).iterdir():
        event_name = event_file.name
        
        # Load text and image regions
        text_region = cv2.imread(str(event_file))
        
        reference_frames[event_name] = {
            'text': text_region,
        }
    
    return reference_frames

def process_single_video(video_path, reference_frames, output_path, threshold, text_weight):
    """
    Process a single video file and save its events to a JSON file
    """
    # Generate output filename based on input video name
    video_name = Path(video_path).stem
    if not video_name.endswith('_events'):
        output_file = Path(output_path) / f"{video_name}_events.json"
    else:
        output_file = Path(output_path) / f"{video_name}.json"

    # Find events in the video
    detected_events = find_events(
        str(video_path),
        reference_frames,
        threshold=threshold,
        text_weight=text_weight
    )

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the detected events
    with open(output_file, 'w') as f:
        json.dump(detected_events, f, indent=2)

    return output_file

def process_video_folder(folder_path, reference_frames, output_path, threshold, text_weight):
    """
    Process all video files in a folder
    """
    processed_files = []
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    
    # Convert to Path object
    folder_path = Path(folder_path)
    
    # Process each video file in the folder
    for video_file in folder_path.iterdir():
        if video_file.suffix.lower() in video_extensions:
            output_file = process_single_video(
                video_file,
                reference_frames,
                output_path,
                threshold,
                text_weight
            )
            processed_files.append(output_file)
    
    return processed_files


def compute_text_similarity(text1, text2, debug=False):
    """
    Compute similarity between text regions using OCR-friendly comparison with empty frame detection
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(text1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(text2, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary
    _, binary1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)  # Inverted threshold
    _, binary2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)  # Inverted threshold
    
    # Calculate percentage of non-white pixels (text pixels)
    text_pixels1 = np.count_nonzero(binary1) / binary1.size
    text_pixels2 = np.count_nonzero(binary2) / binary2.size
    
    # If either image is mostly empty (very few text pixels), return low similarity
    empty_threshold = 0.001  # Adjust this threshold as needed
    if text_pixels1 < empty_threshold or text_pixels2 < empty_threshold:
        similarity = 0.0
    else:
        # Compare using normalized cross-correlation
        result = cv2.matchTemplate(binary1, binary2, cv2.TM_CCORR_NORMED)
        similarity = float(result.max())
    
    if debug:
        # Create debug visualization
        def create_titled_image(title, image):
            # Add title text to image
            h, w = image.shape[:2]
            title_image = np.zeros((60, w), dtype=np.uint8) + 255  # white background
            cv2.putText(title_image, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return np.vstack([title_image, image])
        
        # Create titled versions of each processing stage
        titled_gray1 = create_titled_image("Reference Text (Grayscale)", gray1)
        titled_gray2 = create_titled_image("Current Text (Grayscale)", gray2)
        titled_binary1 = create_titled_image("Reference Text (Binary)", binary1)
        titled_binary2 = create_titled_image("Current Text (Binary)", binary2)
        
        # Stack images horizontally and vertically
        top_row = np.hstack([titled_gray1, titled_gray2])
        bottom_row = np.hstack([titled_binary1, titled_binary2])
        debug_image = np.vstack([top_row, bottom_row])
        
        # Add white background rectangle for score text
        score_text = f"Similarity: {similarity:.3f} (Text%: {text_pixels1:.3%}, {text_pixels2:.3%})"
        (text_width, text_height), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        score_bg = np.zeros((text_height + 20, debug_image.shape[1]), dtype=np.uint8) + 255
        debug_image = np.vstack([debug_image, score_bg])
        
        # Add similarity score with black text on white background
        cv2.putText(debug_image, score_text,
                   (10, debug_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Display debug image
        cv2.imshow("Text Matching Debug", debug_image)
        key = cv2.waitKey(1)  # Update window and continue
        if key == ord('q'):  # Allow quitting with 'q' key
            cv2.destroyAllWindows()
    
    return similarity

def compute_image_similarity(img1, img2):
    """
    Compute similarity between images using histogram comparison
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Compare histograms using correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(correlation)  # Convert to Python float

def match_images(current_images, reference_images):
    """
    Match current frame images with reference images, considering possible swapping
    Returns the maximum similarity score considering both possible arrangements
    """
    # Try both possible arrangements
    sim1 = (compute_image_similarity(current_images[0], reference_images[0]) +
            compute_image_similarity(current_images[1], reference_images[1])) / 2
    
    sim2 = (compute_image_similarity(current_images[0], reference_images[1]) +
            compute_image_similarity(current_images[1], reference_images[0])) / 2
    
    return float(max(sim1, sim2))  # Convert to Python float

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
    left_image = frame[height//3:, 0:width//2]  # Bottom left for left image
    right_image = frame[height//3:, width//2:]  # Bottom right for right image
    
    # Remove timer region (top left corner)
    text_region = text_region.copy()  # Create a copy to avoid modifying the original
    text_region[0:height//10, 0:width//10] = 255  # Make timer region white
    
    return {
        'text': text_region,
        'left_image': left_image,
        'right_image': right_image
    }

def find_events(video_path, reference_frames, threshold=0.8, text_weight=0.7):
    """
    Find events in a new video by comparing text and image regions separately
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    
    current_events = defaultdict(lambda: {'start': None, 'score': 0})
    detected_events = []
    
    frame_number = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        current_time = float(frame_number / fps)  # Convert to Python float
        frame_number += 1
        
        # Extract regions from current frame
        regions = extract_regions(frame)
        current_images = [regions['left_image'], regions['right_image']]
        
        print( current_time)
        if current_time < 30: 
            continue

        debug = current_time > 30.0 and False
        # Compare with each reference event
        for event_name, ref_data in reference_frames.items():
            text_sim = 0.0
            # Compare text regions
            if text_weight > 0.0:
                text_sim = compute_text_similarity(regions['text'], ref_data['text'], debug)
            
            # Compare images (allowing for swapped positions)
            image_sim = 0.0
            if text_weight < 1.0:
                image_sim = match_images(current_images, ref_data['images'])
            
            # Compute weighted similarity score
            total_sim = float(text_weight * text_sim + (1 - text_weight) * image_sim)
            
            if total_sim > threshold:
                if current_events[event_name]['start'] is None:
                    current_events[event_name]['start'] = current_time
                current_events[event_name]['score'] += total_sim
            else:
                if current_events[event_name]['start'] is not None:
                    detected_events.append({
                        'event': event_name,
                        'start_time': float(current_events[event_name]['start']),  # Convert to Python float
                        'end_time': float(current_time),  # Convert to Python float
                        'confidence': float(current_events[event_name]['score'] /  # Convert to Python float
                                    ((current_time - current_events[event_name]['start']) * fps))
                    })
                    current_events[event_name] = {'start': None, 'score': 0}
            
    video.release()
    
    # Sort events by start time
    detected_events.sort(key=lambda x: x['start_time'])
    
    return detected_events

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find events in videos')
    parser.add_argument('input_path', 
                       help='Path to either a video file or a folder containing videos')
    parser.add_argument('reference_dir', 
                       help='Directory containing reference frames')
    parser.add_argument('output_path', 
                       help='Path to save the detected events (JSON)')
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Similarity threshold (default: 0.8)')
    parser.add_argument('--text-weight', type=float, default=0.7,
                      help='Weight given to text similarity vs image similarity (default: 0.7)')
    
    args = parser.parse_args()
    
    # Load reference frames
    reference_frames = load_reference_frames(args.reference_dir)
    
    # Check if input path is a file or directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        # Process single video file
        output_file = process_single_video(
            input_path,
            reference_frames,
            args.output_path,
            args.threshold,
            args.text_weight
        )
        print(f"Processed video and saved events to: {output_file}")
    
    elif input_path.is_dir():
        # Process all videos in the folder
        processed_files = process_video_folder(
            input_path,
            reference_frames,
            args.output_path,
            args.threshold,
            args.text_weight
        )
        print(f"Processed {len(processed_files)} videos:")
        for file in processed_files:
            print(f"  - {file}")
    
    else:
        print(f"Error: Input path '{input_path}' does not exist or is neither a file nor a directory")
        exit(1)

if __name__ == "__main__":
    main()