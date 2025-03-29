"""
This file describes functions used in 'parse_logs.py' and not an 'executable' script.

Given an excel file with stimuli sentenses and a psychopy log file, extract reaction times for phrases from the log.
"""

from extract_phrases import extract_phrases

import sys
import re
import json
import csv
import unicodedata
import pandas as pd

def normalize_text(text):
    """
    Normalize text by removing accents, handling apostrophes, and normalizing spacing.
    
    Args:
        text (str): Text to normalize
    
    Returns:
        str: Normalized text
    """
    # First normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace accented characters with their non-accented versions
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    text = text.replace('ê', 'e')

    # Standardize apostrophes (both ' and ')
    text = text.replace('`', "").replace("'", "")
    
    # Remove other punctuation that might cause issues
    text = re.sub(r'[.,;:!?«»""()]', ' ', text)
    
    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for more robust matching
    text = text.lower()
    
    return text.strip()


def extract_time_for_matching_phrases(log_file_path, target_phrases, phrase_df):
    """
    Extract timestamps for lines where Sentence_Stimuli matches a phrase in the provided list,
    ignoring differences in spacing.
    
    Args:
        log_file_path (str): Path to the log file
        target_phrases (list): List of phrases to match
        phrase_df (DataFrame): DataFrame containing all columns from the input file
    
    Returns:
        dict: Dictionary mapping phrases to their timestamps and additional data
    """
    # Normalize target phrases by replacing multiple spaces with a single space
    normalized_target_phrases = [normalize_text(phrase) for phrase in target_phrases]
    
    # Create a mapping from normalized phrases back to original phrases
    phrase_mapping = {normalize_text(phrase): phrase for phrase in target_phrases}
    
    # Create a phrase to index mapping from original DataFrame
    phrase_to_index = {row['Sentence_Stimuli']: idx for idx, row in phrase_df.iterrows()}
    
    results = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if the line contains 'Sentence_Stimuli'
            if 'Sentence_Stimuli' not in line:
                i += 1
                continue
                
            # Extract the timestamp and phrase
            timestamp = extract_time_from_line(line)
            match = re.search(r"'Sentence_Stimuli':\s*(.*?),\s*'Phrase_type':", line)
            if not match:
                i += 1
                continue
                
            extracted_phrase = match.group(1).strip()
            normalized_extracted = normalize_text(extracted_phrase)
            
            # Check if the normalized phrase is in our target list
            if normalized_extracted not in normalized_target_phrases:
                i += 1
                continue
                
            # Use the original phrase for the results
            original_phrase = phrase_mapping[normalized_extracted]
            
            # Create item data structure
            item = {
                "item_start": timestamp,
                "reaction_time": None,
                "rating": None,
                "original_index": phrase_to_index.get(original_phrase, -1)  # Get original index
            }
            
            # Continue reading subsequent lines to collect more data about this item
            j = i + 1
            while j < len(lines) and 'Sentence_Stimuli' not in lines[j]:
                next_line = lines[j]

                # look for response reaction data 
                if 'Left button down' in next_line:
                    reaction_ts = extract_time_from_line(next_line)
                    item["reaction_time"] = round(reaction_ts - timestamp, 4)
                
                # look for rating
                if 'Target_Choice: rating =' in next_line:
                    rating_match = re.search(r'Target_Choice: rating = ([\d\.]+)', next_line)
                    if rating_match:
                        item["rating"] = int(float(rating_match.group(1)))
                
                j += 1
            
            # Store the results
            if original_phrase not in results:
                results[original_phrase] = item
            
            # Move to the next potential item
            i = j
    
    return results


def extract_time_from_line(line):
    match = re.match(r'^(\d+\.\d+)', line.strip())
    if match:
        return float(match.group(1))
    return None


from extract_phrases import extract_phrases

import sys
import re
import json
import csv
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Normalize text by removing accents, handling apostrophes, and normalizing spacing.
    
    Args:
        text (str): Text to normalize
    
    Returns:
        str: Normalized text
    """
    # First normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace accented characters with their non-accented versions
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    text = text.replace('ê', 'e')

    # Standardize apostrophes (both ' and ')
    text = text.replace('`', "").replace("'", "")
    
    # Remove other punctuation that might cause issues
    text = re.sub(r'[.,;:!?«»""()]', ' ', text)
    
    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for more robust matching
    text = text.lower()
    
    return text.strip()


def extract_time_for_matching_phrases(log_file_path, target_phrases, phrase_df):
    """
    Extract timestamps for lines where Sentence_Stimuli matches a phrase in the provided list,
    ignoring differences in spacing.
    
    Args:
        log_file_path (str): Path to the log file
        target_phrases (list): List of phrases to match
        phrase_df (DataFrame): DataFrame containing all columns from the input file
    
    Returns:
        dict: Dictionary mapping phrases to their timestamps and additional data
    """
    # Normalize target phrases by replacing multiple spaces with a single space
    normalized_target_phrases = [normalize_text(phrase) for phrase in target_phrases]
    
    # Create a mapping from normalized phrases back to original phrases
    phrase_mapping = {normalize_text(phrase): phrase for phrase in target_phrases}
    
    # Create a phrase to index mapping from original DataFrame
    phrase_to_index = {row['Sentence_Stimuli']: idx for idx, row in phrase_df.iterrows()}
    
    results = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if the line contains 'Sentence_Stimuli'
            if 'Sentence_Stimuli' not in line:
                i += 1
                continue
                
            # Extract the timestamp and phrase
            timestamp = extract_time_from_line(line)
            match = re.search(r"'Sentence_Stimuli':\s*(.*?),\s*'Phrase_type':", line)
            if not match:
                i += 1
                continue
                
            extracted_phrase = match.group(1).strip()
            normalized_extracted = normalize_text(extracted_phrase)
            
            # Check if the normalized phrase is in our target list
            if normalized_extracted not in normalized_target_phrases:
                i += 1
                continue
                
            # Use the original phrase for the results
            original_phrase = phrase_mapping[normalized_extracted]
            
            # Create item data structure
            item = {
                "item_start": timestamp,
                "reaction_time": None,
                "rating": None,
                "original_index": phrase_to_index.get(original_phrase, -1)  # Get original index
            }
            
            # Continue reading subsequent lines to collect more data about this item
            j = i + 1
            while j < len(lines) and 'Sentence_Stimuli' not in lines[j]:
                next_line = lines[j]

                # look for response reaction data 
                if 'Left button down' in next_line:
                    reaction_ts = extract_time_from_line(next_line)
                    item["reaction_time"] = round(reaction_ts - timestamp, 4)
                
                # look for rating
                if 'Target_Choice: rating =' in next_line:
                    rating_match = re.search(r'Target_Choice: rating = ([\d\.]+)', next_line)
                    if rating_match:
                        item["rating"] = int(float(rating_match.group(1)))
                
                j += 1
            
            # Store the results
            if original_phrase not in results:
                results[original_phrase] = item
            
            # Move to the next potential item
            i = j
    
    return results


def extract_time_from_line(line):
    match = re.match(r'^(\d+\.\d+)', line.strip())
    if match:
        return float(match.group(1))
    return None


def save_results_to_csv(results, output_file, phrase_df):
    """
    Save the results to a CSV file with all columns from the original input file plus
    additional columns: sample_index, phrase_index, reaction_time, rating
    
    Args:
        results (dict): Dictionary mapping phrases to their data
        output_file (str): Path to the output CSV file
        phrase_df (DataFrame): DataFrame containing all columns from the input file
    """
    # Create output DataFrame
    output_rows = []
    
    # Add sample index (simple increment)
    for i, (phrase, data) in enumerate(results.items(), 1):
        # Get the row from the original dataframe for this phrase
        orig_row = phrase_df[phrase_df['Sentence_Stimuli'] == phrase]
        
        if not orig_row.empty:
            # Convert the original row to a dictionary
            row_dict = orig_row.iloc[0].to_dict()
            
            # Add the additional columns
            row_dict.update({
                'sample_index': i,
                'phrase_index': data['original_index'],
                'reaction_time': data['reaction_time'],
                'rating': data['rating']
            })
            
            output_rows.append(row_dict)
    
    # Create DataFrame from the output rows
    output_df = pd.DataFrame(output_rows)
    
    # Reorder columns to ensure sample_index and phrase_index come first
    # Get all column names
    all_columns = list(output_df.columns)
    
    # Remove the columns we want to place at the beginning
    for col in ['sample_index', 'phrase_index']:
        if col in all_columns:
            all_columns.remove(col)
    
    # Create new column order with sample_index and phrase_index first
    new_column_order = ['sample_index', 'phrase_index'] + all_columns
    
    # Reorder the DataFrame columns
    output_df = output_df[new_column_order]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Results saved to {output_file}")
    
def main():
    phrase_path = sys.argv[1]  # put with your actual file path
    log_path = sys.argv[2]
    
    # Load phrases and the original dataframe
    phrase_df = pd.read_excel(phrase_path)
    phrases = extract_phrases(phrase_path)
    
    print(phrases)
    
    result = extract_time_for_matching_phrases(log_path, phrases, phrase_df)
    
    print("Timestamps for matching phrases:")
    for sentence, res in result.items():
        print(f"Phrase: {sentence}")
        print(f"Result: {res}")
        print()

    save_results_to_csv(result, "EN_parsed_log.csv", phrase_df)

if __name__ == "__main__":
    main()