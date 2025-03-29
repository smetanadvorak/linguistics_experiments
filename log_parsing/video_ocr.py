import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import os
import time
from tqdm import tqdm
from pathlib import Path

def similarity_ratio(a, b):
    """Calculate string similarity ratio using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()

def extract_regions(frame):
    """
    Extract different regions from the frame:
    - text region (top)
    Assumes consistent layout across all frames
    """
    height, width = frame.shape[:2]
    
    # Define regions (adjust these based on your exact layout)
    text_region = frame[0:height//3, :]  # Top third for text
    
    # Fill with white to remove timer
    timer_mask_white = np.ones_like(text_region[0:height//10, 0:width//10]) * 255
    text_region[0:height//10, 0:width//10] = timer_mask_white
    
    return text_region

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get black and white image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Invert the image for white text on black background
    thresh = cv2.bitwise_not(thresh)
    
    return thresh

def extract_text_from_frame(frame):
    """Extract text from video frame using OCR."""
    # Extract text region
    text_region = extract_regions(frame)
    
    # Preprocess the image
    processed_frame = preprocess_image(text_region)
    
    # Use pytesseract to extract text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_frame, config=custom_config)
    
    # Clean up text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    return text, text_region

def match_phrase(ocr_text, phrases, threshold=0.80):
    """Match extracted text with phrases from the list."""
    best_match = None
    best_ratio = 0
    
    for phrase in phrases:
        ratio = similarity_ratio(ocr_text.lower(), phrase.lower())
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = phrase
    
    return best_match, best_ratio

def main(video_path, excel_path, output_path, similarity_threshold=0.80, save_images=True, images_dir="text_images"):
    # Load phrases from Excel file
    df = pd.read_excel(excel_path)
    phrases = df['Sentence_Stimuli'].tolist()
    
    # Clean phrases
    phrases = [re.sub(r'\s+', ' ', str(phrase).strip()) for phrase in phrases]
    
    # Create directory for saving text region images
    if save_images:
        Path(images_dir).mkdir(exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Video duration: {duration:.2f} seconds")
    
    # Dictionary to track phrase appearances
    phrase_appearances = {phrase: {'start_frame': None, 'end_frame': None, 
                                   'start_time': None, 'end_time': None, 
                                   'match_quality': 0,
                                   'image_path': None} for phrase in phrases}
    
    # Current phrase being displayed
    current_phrase = None
    
    # Process all frames
    print("Processing frames...")
    frame_idx = 0
    
    # Create progress bar
    pbar = tqdm(total=frame_count)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract text from frame
        frame_text, text_region = extract_text_from_frame(frame)
        
        # Skip empty text
        if frame_text.strip():
            # Match with phrases
            matched_phrase, match_ratio = match_phrase(frame_text, phrases, similarity_threshold)
            
            # If we found a match
            if matched_phrase:
                # If this is a new phrase
                if current_phrase != matched_phrase:
                    # If we were tracking a previous phrase, mark its end
                    if current_phrase and phrase_appearances[current_phrase]['end_frame'] is None:
                        phrase_appearances[current_phrase]['end_frame'] = frame_idx - 1
                        phrase_appearances[current_phrase]['end_time'] = (frame_idx - 1) / fps
                    
                    # Start tracking new phrase
                    current_phrase = matched_phrase
                    
                    # Only update start if not already set
                    if phrase_appearances[current_phrase]['start_frame'] is None:
                        phrase_appearances[current_phrase]['start_frame'] = frame_idx
                        phrase_appearances[current_phrase]['start_time'] = frame_idx / fps
                        
                        # Save image of text region
                        if save_images:
                            # Create a valid directory name from the phrase
                            phrase_dir = re.sub(r'[^\w\s-]', '', current_phrase)[:50]
                            phrase_dir = re.sub(r'\s+', '_', phrase_dir.strip())
                            
                            # Save the image
                            img_path = os.path.join(images_dir, f"{phrase_dir.lower()}.jpg")
                            cv2.imwrite(img_path, text_region)
                            print("Saved image to", img_path)
                            
                            # Store the path
                            phrase_appearances[current_phrase]['image_path'] = img_path
                    
                    # Update match quality if better
                    if match_ratio > phrase_appearances[current_phrase]['match_quality']:
                        phrase_appearances[current_phrase]['match_quality'] = match_ratio
                    
                    print(f"Frame {frame_idx}: Detected phrase: {current_phrase} (Match: {match_ratio:.2f})")
            else:
                # If no match but we were tracking a phrase, mark its end
                if current_phrase and phrase_appearances[current_phrase]['end_frame'] is None:
                    phrase_appearances[current_phrase]['end_frame'] = frame_idx - 1
                    phrase_appearances[current_phrase]['end_time'] = (frame_idx - 1) / fps
                    current_phrase = None
        
        # Increment frame index
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    # Close any unfinished phrase at the end of the video
    if current_phrase and phrase_appearances[current_phrase]['end_frame'] is None:
        phrase_appearances[current_phrase]['end_frame'] = frame_idx - 1
        phrase_appearances[current_phrase]['end_time'] = (frame_idx - 1) / fps
    
    # Create results DataFrame
    results = []
    for phrase, data in phrase_appearances.items():
        # Check if phrase was detected
        if data['start_frame'] is not None:
            results.append({
                'phrase': phrase,
                'start_frame': data['start_frame'],
                'end_frame': data['end_frame'],
                'start_time': data['start_time'],
                'end_time': data['end_time'],
                'duration_seconds': data['end_time'] - data['start_time'] if data['end_time'] is not None else None,
                'match_quality': data['match_quality'],
                'image_path': data['image_path']
            })
    
    # Create DataFrame and save to Excel
    result_df = pd.DataFrame(results)
    
    # Merge with original data
    merged_df = pd.merge(
        df, 
        result_df, 
        left_on='Sentence_Stimuli', 
        right_on='phrase', 
        how='left'
    )
    
    # Save results
    merged_df.to_excel(output_path, index=False)
    
    # Print summary
    detected_count = sum(1 for p in phrase_appearances.values() if p['start_frame'] is not None)
    print(f"\nProcessing complete!")
    print(f"Detected {detected_count} out of {len(phrases)} phrases")
    print(f"Results saved to {output_path}")
    
    # Release video
    cap.release()

if __name__ == "__main__":
    # Set paths
    video_path = "data2/videos_fr/Copy of G10122.mkv"  # Replace with your video path
    excel_path = "data2/Copy of Stimuli French.xlsx"  # Replace with your Excel file path
    output_path = "results/video_ocr_results.xlsx"
    images_dir = "data2/frames_fr"  # Directory to save text region images
    
    # Set pytesseract path if needed (uncomment and modify if necessary)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # macOS or Linux
    
    # Set similarity threshold
    similarity_threshold = 0.75  # Adjust as needed
    
    # Set whether to save text region images
    save_images = True
    
    # Run the main function
    main(video_path, excel_path, output_path, similarity_threshold, save_images, images_dir)