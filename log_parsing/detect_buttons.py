import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
import argparse

from matplotlib import pyplot as plt

def get_frame_at_timestamp(cap, timestamp, fps):
    """Get frame at specific timestamp from an already open video capture."""
    # Calculate frame number from timestamp
    frame_number = int(timestamp * fps) - 1
    
    # Ensure frame number is valid
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= frame_count:
        frame_number = frame_count - 1
        
    # Set position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        raise Exception(f"Failed to extract frame at {timestamp}s")
    
    return frame

def define_button_rois(frame):
    """Define regions of interest for the five buttons at the bottom of the screen."""
    height, width = frame.shape[:2]
    
    # Estimate button positions based on the image
    # These are approximate and may need adjustment
    button_y = int(height * 0.64)  # Vertical position of buttons
    button_height = int(height * 0.1)  # Height of button area
    
    # Define width of each button region and horizontal positions
    button_width = int(width * 0.125)
    
    # Calculate positions for the 5 buttons
    button_positions = []
    for i in range(5):
        button_x = int(width * (0.175 + i * 0.137))  # Distribute buttons horizontally
        button_positions.append((button_x, button_y, button_width, button_height))
    
    return button_positions

def detect_button_press(frame, button_rois):
    """Detect which button was pressed by analyzing pixel values in ROIs."""
    # Convert to grayscale for simpler analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)
    # plt.show()

    # Calculate pixel intensity changes for each button ROI
    button_scores = []
    button_scores2 = []
    for x, y, w, h in button_rois:
        roi = gray[y:y+h, x:x+w]
        
        # Calculate standard deviation of pixel values as a measure of "activity"
        # Higher std dev may indicate presence of a cursor
        score = np.sum(roi < 30)
        button_scores.append(score)

        score2 = np.sum((roi > 115) & (roi < 130))
        button_scores2.append(score2)

    
    # The button with highest score is likely the one with mouse activity
    button_scores = np.array(button_scores)
    valid_indices1 = np.argmax(button_scores)

    button_scores2 = np.array(button_scores2)
    valid_indices2 = np.where((button_scores2 < 1500))[0]

    valid_indices = set()
    valid_indices.update((valid_indices1,))
    valid_indices.union(set(valid_indices2.tolist()))

    # Check if we have exactly one valid index
    if len(valid_indices) == 0:
        print(button_scores, button_scores2)
        print(valid_indices)
        print("Weird")
        return None
    elif len(valid_indices) > 1:
        print(button_scores, button_scores2)
        print(valid_indices)
        print("Weird")
        return None
    else:
        # We have exactly one valid index
        pressed_button_idx = list(valid_indices)[0]

    # if button_scores[pressed_button_idx] > 1975:
    #     print(button_scores)
    #     print("Weird")
    #     return None
    
    # Add 1 to convert from 0-based to 1-based indexing
    return pressed_button_idx + 1

def format_event_name(event_name):
    """Convert event name from filename format to readable format."""
    # Remove .jpg extension if present
    if event_name.endswith('.jpg'):
        event_name = event_name[:-4]
    
    # Replace underscores with spaces
    event_name = event_name.replace('_', ' ')
    
    # Replace escape sequences with actual characters
    event_name = event_name.replace('\\u00e9', 'é')
    event_name = event_name.replace('\\u00e8', 'è')
    event_name = event_name.replace('\\u00ea', 'ê')
    
    return event_name

def analyze_events(json_folder, video_folder, output_folder):
    """Analyze all events in JSON files and detect button presses."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all JSON files in the folder
    json_files = list(Path(json_folder).glob('*.json'))
    
    for json_path in json_files:
        # Determine corresponding video file
        video_filename = f"{json_path.stem[:-7]}.mkv"
        video_path = Path(video_folder) / video_filename
        
        if not video_path.exists():
            print(f"Warning: Video file {video_path} not found for {json_path}")
            continue
            
        print(f"Processing {json_path.name} with {video_path.name}")
        
        # Load events from JSON
        with open(json_path, 'r') as f:
            events_data = json.load(f)
        
        # Create a CSV file for this video
        csv_filename = f"{json_path.stem}_results.csv"
        csv_path = Path(output_folder) / csv_filename
        
        # Open video file once
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Open CSV file for writing
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['event_name', 'start', 'end', 'duration', 'response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process each event
            for event in events_data:
                event_name = event['event']
                start_time = event['start_time']
                end_time = event['end_time']
                duration = end_time - start_time
                
                # try:
                # Extract the frame at the end time
                frame = get_frame_at_timestamp(cap, end_time, fps)
                
                # Define button ROIs
                button_rois = define_button_rois(frame)
                
                # Detect which button was pressed
                button_number = detect_button_press(frame, button_rois)
                if button_number is None:
                    output_path = os.path.join("debug_output", f"roi_visualization_{event_name}.jpg")
                    visualize_button_detection(frame, output_path)
                    exit()
                
                # Format event name
                formatted_event_name = format_event_name(event_name)
                
                # Write result to CSV
                writer.writerow({
                    'event_name': formatted_event_name,
                    'start': start_time,
                    'end': end_time,
                    'duration': round(duration, 3),
                    'response': button_number
                })
                
                print(f"  Event: {formatted_event_name}")
                print(f"    Button {button_number} pressed at {end_time:.2f}s (duration: {duration:.2f}s)")
                    
                # except Exception as e:
                #     print(f"  Error processing event {event_name}: {e}")
        
        # Close video file
        cap.release()
        
        print(f"Results for {json_path.stem} saved to {csv_path}")

def visualize_button_detection(frame, output_path=None):
    """Generate a visualization of the button detection for debugging."""
    # Make a copy of the frame to avoid modifying the original
    vis_frame = frame.copy()
    
    # Define button ROIs
    button_rois = define_button_rois(frame)
    
    # Draw rectangles around button areas
    for i, (x, y, w, h) in enumerate(button_rois):
        color = (0, 255, 0)  # Green rectangle
        thickness = 2
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(vis_frame, f"Button {i+1}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Calculate scores for each button
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = []
    for x, y, w, h in button_rois:
        roi = gray[y:y+h, x:x+w]
        score = np.sum(roi < 150)
        scores.append(score)
    
    # Display scores on the visualization
    for i, score in enumerate(scores):
        x, y = button_rois[i][0], button_rois[i][1]
        cv2.putText(vis_frame, f"Score: {score:.2f}", (x, y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save the visualization if output path is provided
    if output_path:
        cv2.imwrite(output_path, vis_frame)
        print(f"Visualization saved to {output_path}")
    
    # Detect which button was pressed
    button_number = detect_button_press(frame, button_rois)
    print(f"Detected press of Button {button_number} (scores: {[f'{s:.2f}' for s in scores]})")
    
    return vis_frame

def debug_roi_visualization():
    """Debug function to visualize ROIs on a sample frame."""
    # This function can be commented out after fine-tuning the ROIs
    print("Debugging ROI visualization...")
    
    # Choose a sample video and timestamp
    sample_video = "data2/log_parsing/data2/videos_fr/Copy of G10122.mkv"
    sample_timestamp = 81.1  # Adjust based on your data
    
    cap = cv2.VideoCapture(sample_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {sample_video}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame = get_frame_at_timestamp(cap, sample_timestamp, fps)
    cap.release()
    
    # Create output folder if it doesn't exist
    os.makedirs("debug_output", exist_ok=True)
    
    # Visualize ROIs
    output_path = os.path.join("debug_output", "roi_visualization.jpg")
    visualize_button_detection(frame, output_path)
    
    print("ROI visualization complete. Check the debug_output folder.")

def main():
    parser = argparse.ArgumentParser(description='Analyze button presses in video events')
    parser.add_argument('--json_folder', required=True, help='Path to folder containing event JSON files')
    parser.add_argument('--video_folder', required=True, help='Path to folder containing video files')
    parser.add_argument('--output_folder', default='results', help='Output folder for CSV files')
    parser.add_argument('--debug_roi', action='store_true', help='Debug ROI visualization')
    
    args = parser.parse_args()
    
    # Uncomment the following line to debug ROIs
    # debug_roi_visualization()
    
    if args.debug_roi:
        print("ROI debugging mode")
        debug_json = list(Path(args.json_folder).glob('*.json'))[0]
        debug_video = Path(args.video_folder) / f"{debug_json.stem[:-7] if '_events' in debug_json.stem else debug_json.stem}.mkv"
        
        with open(debug_json, 'r') as f:
            events_data = json.load(f)
            
        if events_data:
            # Create debug output folder if it doesn't exist
            os.makedirs("debug_output", exist_ok=True)
            
            # Open video file once
            cap = cv2.VideoCapture(str(debug_video))
            if not cap.isOpened():
                print(f"Error: Could not open video {debug_video}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Process a few events for visualization
            sample_size = min(3, len(events_data))
            for i in range(sample_size):
                event = events_data[i]
                timestamp = event['end_time']
                
                # Get frame at timestamp
                frame = get_frame_at_timestamp(cap, timestamp, fps)
                
                # Visualize button detection
                output_path = os.path.join("debug_output", f"event_{i}_time_{timestamp:.2f}.jpg")
                visualize_button_detection(frame, output_path)
                
            cap.release()
    else:
        # Normal operation
        analyze_events(args.json_folder, args.video_folder, args.output_folder)

if __name__ == "__main__":
    main()