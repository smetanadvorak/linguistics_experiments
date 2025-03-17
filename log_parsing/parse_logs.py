import os
import sys
import re
from extract_phrases import extract_phrases
from log_parser import extract_time_for_matching_phrases, save_results_to_csv, normalize_text

def process_log_files(phrase_path, log_folder):
    """
    Process all log files in the specified folder and save results as CSV files.
    
    Args:
        phrase_path (str): Path to the file containing phrases
        log_folder (str): Path to the folder containing log files
    """
    # Load phrases
    phrases = extract_phrases(phrase_path)
    print(f"Loaded {len(phrases)} phrases from {phrase_path}")

    # Create output directory if it doesn't exist
    output_dir = "log_parsing_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all files in the log folder
    log_files = [f for f in os.listdir(log_folder) if os.path.isfile(os.path.join(log_folder, f))]
    print(f"Found {len(log_files)} files in {log_folder}")
    
    # Process each log file
    for log_file in log_files:
        log_file_path = os.path.join(log_folder, log_file)
        
        # Skip non-text files (optional)
        if not log_file.endswith('.txt') and not log_file.endswith('.log'):
            print(f"Skipping non-log file: {log_file}")
            continue
        
        print(f"Processing {log_file}...")
        
        # Extract data from the log file
        try:
            results = extract_time_for_matching_phrases(log_file_path, phrases)

            # diff = [p for p in phrases if p not in results]
            # [print(d, '\n', normalize_text(d)) for d in diff]
            # exit()

            # if len(results) != 168:
            #     raise Exception(f"Got {len(results)} items instead of exactly 168")
            # Create output filename
            output_file = os.path.join(output_dir, f"{os.path.splitext(log_file)[0]}.csv")   

            # Save results to CSV
            print(output_file)
            save_results_to_csv(results, output_file)
            # print(f"  - Saved results to {output_file}")
            
        except Exception as e:
            print(f"  - Error processing {log_file}: {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_processor.py phrase_file log_folder")
        sys.exit(1)
        
    phrase_path = sys.argv[1]  # path to file with phrases
    log_folder = sys.argv[2]   # path to folder with log files
    
    process_log_files(phrase_path, log_folder)
    print("Processing complete!")

if __name__ == "__main__":
    main()