import argparse
import os
import glob
import json
from pathlib import Path

from aoi import *

def extract_id(filename):
    """
    Extract participant ID from filename.
    Works with both formats:
    - G10110_SPM.csv
    - G10110-scrrec_events.json
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Extracted ID (e.g., 'G10110')
    """
    # Get just the filename without path
    base_name = os.path.basename(filename)
    # Split by both possible separators
    parts = base_name.replace('-', '_').split('_')
    # First part should be the ID in both cases
    return parts[0]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process an input file and three folders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add required input file argument
    parser.add_argument(
        '--template',
        required=True,
        help='Path to the template aoi xml file'
    )
    
    # Add three required folder arguments
    parser.add_argument(
        '--psycho',
        required=True,
        help='Path to the folder with psychopi csvs'
    )
    parser.add_argument(
        '--events',
        required=True,
        help='Path to the folder with event jsons'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to the output folder'
    )
    
    args = parser.parse_args()
    
    # Verify that input file exists
    if not os.path.isfile(args.template):
        parser.error(f'Input file does not exist: {args.template}')
    
    # Verify that all folders exist
    for folder in [args.psycho, args.events, args.output]:
        if not os.path.isdir(folder):
            parser.error(f'Folder does not exist: {folder}')
    
    return args

def main():
    args = parse_arguments()
    
    # Get all psycho CSV files
    psycho_files = glob.glob(os.path.join(args.psycho, '*_SPM.csv'))
    
    # Keep track of unmatched files
    unmatched_psycho = []
    
    # Process each psycho file
    for psycho_path in psycho_files:
        # Extract ID using the common function
        psycho_id = extract_id(psycho_path)
        
        # Construct the expected events filename
        events_filename = f"{psycho_id}-scrrec_events.json"
        events_path = os.path.join(args.events, events_filename)
        
        # Check if matching events file exists
        if not os.path.exists(events_path):
            print(f"Warning: No matching events file found for {psycho_path}")
            unmatched_psycho.append(psycho_path)
            continue
            
        # Here both files exist and match - process the pair
        print(f"Processing pair for ID {psycho_id}:")
        print(f"  Psycho file: {psycho_path}")
        print(f"  Events file: {events_path}")
        
        with open(events_path, 'r') as f:
            events_data = json.load(f)
            events_dict = {event['event']: event for event in events_data}
        
        template_xml = XMLProcessor(args.template)
        aoi_list, item_dict = template_xml.read_xml()
        sentences_df = pd.read_csv(psycho_path)

        ordered_aoi_list = template_xml.reorder_by_sentences2(item_dict, sentences_df, events_dict)
        output_path = os.path.join(args.output, psycho_id + "_aoi.xml")
        template_xml.write_xml(ordered_aoi_list, output_path)

        
    # Report unmatched files at the end
    if unmatched_psycho:
        print("\nUnmatched psycho files:")
        for path in unmatched_psycho:
            print(f"  {path}")

if __name__ == '__main__':
    main()


'''
aoi_batch.py \
    --template /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/en/Example_G20406EN-scrrec-AOIs.xml \
    --psycho /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr/psycho_py \
    --events /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr/screen_videos/output_annotation \
    --output /Users/akmbpro/Documents/coding/alina/output_scripts/data/ex1/fr/output_aoi  

'''