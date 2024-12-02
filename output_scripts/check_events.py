import json
import os
from typing import Set, Dict, List

def load_reference_events() -> Set[str]:
    """Load the reference set of events that should be present in all files."""
    reference_events = {
        "anim_text.jpg", "glass_text.jpg", "girl_text.jpg", "book_text.jpg",
        "ball_text.jpg", "pat_text.jpg", "game_text.jpg", "cat_text.jpg",
        "trol_text.jpg", "bird_text.jpg", "pic_text.jpg", "dog_text.jpg"
    }
    return reference_events

def check_json_file(filepath: str, reference_events: Set[str]) -> Dict[str, List[str]]:
    """Check a single JSON file for missing events."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print(f"Error: {filepath} does not contain a list of events")
            return {'missing': list(reference_events), 'extra': []}
            
        file_events = {item.get('event', '') for item in data}
        
        missing_events = reference_events - file_events
        extra_events = file_events - reference_events
        
        return {
            'missing': sorted(list(missing_events)),
            'extra': sorted(list(extra_events))
        }
        
    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file")
        return {'missing': list(reference_events), 'extra': []}
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return {'missing': list(reference_events), 'extra': []}

def check_folder(folder_path: str) -> None:
    """Check all JSON files in the specified folder for missing events."""
    reference_events = load_reference_events()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
        
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
        
    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        result = check_json_file(filepath, reference_events)
        
        if result['missing'] or result['extra']:
            print(f"\n{filename}:")
            if result['missing']:
                print("Missing events:")
                for event in result['missing']:
                    print(f"  - {event}")
            if result['extra']:
                print("Extra events:")
                for event in result['extra']:
                    print(f"  - {event}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    check_folder(folder_path)