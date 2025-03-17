from extract_phrases import extract_phrases

import sys
import re
import json
import csv
import unicodedata


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


def extract_time_for_matching_phrases(log_file_path, target_phrases):
    """
    Extract timestamps for lines where Sentence_Stimuli matches a phrase in the provided list,
    ignoring differences in spacing.
    
    Args:
        log_file_path (str): Path to the log file
        target_phrases (list): List of phrases to match
    
    Returns:
        dict: Dictionary mapping phrases to their timestamps and additional data
    """
    # Normalize target phrases by replacing multiple spaces with a single space
    normalized_target_phrases = [normalize_text(phrase) for phrase in target_phrases]

    # [print(phrase, ' -> ', normalize_text(phrase)) for phrase in target_phrases]
    # exit()
    
    # Create a mapping from normalized phrases back to original phrases
    phrase_mapping = {normalize_text(phrase): phrase for phrase in target_phrases}
    
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
            # match = re.search(r"'Sentence_Stimuli':\s*'(.*?)'", line)
            # match = re.search(r"'Sentence_Stimuli':\s*['\"]([^'\"]*)['\"]", line)
            match = re.search(r"'Sentence_Stimuli':\s*(.*?),\s*'Phrase_type':", line)
            if not match:
                i += 1
                continue
                
            extracted_phrase = match.group(1).strip()
            normalized_extracted = normalize_text(extracted_phrase)
            
            # if 'taureau' in extracted_phrase:  # and 'difficilement' in extracted_phrase:
            #     print(extracted_phrase)
            #     print(normalized_extracted)
            #     print(normalized_extracted in normalized_target_phrases)
            #     input("Continue")

            # Check if the normalized phrase is in our target list
            if normalized_extracted not in normalized_target_phrases:
                # print(f"WARNING: phrase {normalized_extracted} didn't match known phrases")
                i += 1
                continue
                
            # Use the original phrase for the results
            original_phrase = phrase_mapping[normalized_extracted]
            
            # Create item data structure
            item = {
                "item_start": timestamp,
                "reaction_time": None,
                "rating": None
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


def save_results_to_csv(results, output_file):
    """
    Save the results to a CSV file with columns: phrase, reaction_time, rating
    
    Args:
        results (dict): Dictionary mapping phrases to their data
        output_file (str): Path to the output CSV file
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['phrase', 'reaction_time', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for phrase, data in results.items():
            writer.writerow({
                'phrase': phrase,
                'reaction_time': data['reaction_time'],
                'rating': data['rating']
            })
    
    print(f"Results saved to {output_file}")
    
def main():
    phrase_path = sys.argv[1]  # put with your actual file path
    log_path = sys.argv[2]
    
    phrases = extract_phrases(phrase_path)
    print(phrases)
    
    result = extract_time_for_matching_phrases(log_path, phrases)
    
    print("Timestamps for matching phrases:")
    for sentence, res in result.items():
        print(f"Phrase: {sentence}")
        print(f"Result: {res}")
        print()

    save_results_to_csv(result, "EN_parsed_log.xlsx")

if __name__ == "__main__":
    main()