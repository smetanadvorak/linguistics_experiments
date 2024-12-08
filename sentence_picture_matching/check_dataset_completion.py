import os
import sys
from pathlib import Path

def normalize_id(file_id, language):
    """Remove _SPM suffix and language suffix if present"""
    # Remove _SPM suffix
    file_id = file_id.replace('_SPM', '')
    # Remove language suffix if it matches current language
    lang_upper = language.upper()
    if file_id.endswith(lang_upper):
        file_id = file_id[:-len(lang_upper)]
    return file_id

def get_file_ids(directory, extension, language):
    """Extract IDs from files with given extension in the directory"""
    try:
        if not os.path.exists(directory):
            return set()
        files = list(Path(directory).glob(f'*{extension}'))
        return {normalize_id(f.stem.split('-')[0], language) for f in files}
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")
        return set()

def check_completion(base_path, language):
    """Check completion status for a specific language folder and print missing files"""
    print(f"\n=== Checking {language.upper()} dataset ===")
    
    # Define paths for each type of file
    lang_path = os.path.join(base_path, 'ex1', language)
    if not os.path.exists(lang_path):
        print(f"Language directory not found: {lang_path}")
        return

    input_videos_path = os.path.join(lang_path, 'screen_videos', 'input_videos')
    psycho_py_path = os.path.join(lang_path, 'psycho_py')
    events_path = os.path.join(lang_path, 'screen_videos', 'output_annotation')
    aoi_path = os.path.join(lang_path, 'output_aoi')
    
    # Get IDs from each directory
    video_ids = get_file_ids(input_videos_path, '.mkv', language)
    psycho_ids = get_file_ids(psycho_py_path, '.csv', language)
    event_ids = get_file_ids(events_path, '.json', language)
    aoi_ids = get_file_ids(aoi_path, '.xml', language)
    
    # Get all unique IDs
    all_ids = video_ids | psycho_ids | event_ids | aoi_ids
    
    if not all_ids:
        print("No files found")
        return

    # Check for missing files
    has_missing = False
    for file_id in sorted(all_ids):
        missing_types = []
        
        if file_id not in video_ids:
            missing_types.append("input video")
        if file_id not in psycho_ids:
            missing_types.append("psycho data")
        if file_id not in event_ids:
            missing_types.append("events file")
        # if file_id not in aoi_ids:
        #     missing_types.append("AOI output")
            
        if missing_types:
            has_missing = True
            print(f"ID {file_id} missing: {', '.join(missing_types)}")
    
    if not has_missing:
        print("All files are present")

def check_all_languages(base_path):
    """Check completion for all languages"""
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
        
    for language in ['en', 'fr', 'ru']:
        check_completion(base_path, language)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_dataset_completion.py <data_directory>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    check_all_languages(base_path)