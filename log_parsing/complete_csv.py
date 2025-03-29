""" Given a csv of format: 
event_name,start,end,duration,response
this girl is anxious,48.4,50.3,1.9,5
this problem is unsolvable,52.3,53.6,1.3,5
this hedgehog is easy for rolling,55.6,58.2,2.6,5
it is delicious to eat this pizza,60.2,63.1,2.9,4
this ball is bounceable,65.1,67.2,2.1,5
...

and a template of format:
Sentence_Stimuli	Phrase_type	Verb_type_trans	Verb_subtype_trans	Animacy	Anymacy_subtype
This  celebrity    is    easy     to recognise. 	0	1	3	1	2
You    are   easy   to recognise  this celebrity.	1	1	3	1	2
This celebrity    is   easily    recognised.	2	1	3	1	2
This celebrity    is  of  easy    recognition.	3	1	3	1	2
This celebrity    is  easy    for recognition.	4	1	3	1	2
This celebrity  is  recognisable. 	5	1	3	1	2
This    cat      is  difficult  to wash.       	0	1	2	1	3
...

complete the phrases in the initial csv with:
- index of phrase (column 1);
- index of phrase at which it appears in the template csv (column 2);
- all attributes of the phrase from the template (columns 3..)

take path to template and path to folder with initial csv files, run for each initial file independently;
take path to output folder and save results there with suffix '_complete'


python3 complete_csv.py /Users/akmbpro/Documents/coding/alina/log_parsing/data2/Copy\ of\ Stimuli\ English.xlsx /Users/akmbpro/Documents/coding/alina/log_parsing/data2/buttons_en data2/complete_en

python3 complete_csv.py /Users/akmbpro/Documents/coding/alina/log_parsing/data2/Copy\ of\ Stimuli\ French.xlsx /Users/akmbpro/Documents/coding/alina/log_parsing/data2/buttons_fr data2/complete_fr

"""
import os
import pandas as pd
import argparse
import re
import Levenshtein
from tqdm import tqdm

def normalize_sentence(sentence):
    """Normalize a sentence for better matching by removing punctuation, converting to lowercase,
    removing leading/trailing spaces, and replacing multiple spaces with a single space"""
    # Convert to lowercase and remove punctuation
    normalized = re.sub(r'[^\w\s]', '', str(sentence).lower())
    # Remove leading/trailing spaces and replace multiple spaces with one
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def extract_main_phrase(row):
    """Extract the main phrase from an event_name field"""
    # For the initial CSV format, the event_name field contains the sentence
    try:
        return normalize_sentence(row['event_name'])
    except KeyError:
        # If 'event_name' doesn't exist, try alternative column names
        for col in ['event', 'name', 'sentence', 'text', 'phrase']:
            if col in row:
                return normalize_sentence(row[col])
        
        # If no known column name works, use the first column
        return normalize_sentence(row.iloc[0])

def extract_template_phrase(row):
    """Extract the main phrase from a template row"""
    # Get the sentence from the first column (typically Sentence_Stimuli)
    # We need to handle different ways of accessing the column based on how pandas reads Excel
    try:
        # Try by first column name
        first_col_name = row.index[0]
        return normalize_sentence(row[first_col_name])
    except (IndexError, KeyError, AttributeError):
        # If that fails, try positional access
        try:
            return normalize_sentence(row.iloc[0])
        except (IndexError, AttributeError):
            # Last resort, try numeric index
            try:
                return normalize_sentence(row[0])
            except (IndexError, KeyError):
                # If everything fails, raise an error
                raise ValueError("Could not extract template phrase from row")

def find_best_match(phrase, template_df, threshold=0.2, ambiguity_threshold=2):
    """
    Find the best matching template row for a given phrase using edit distance
    
    Args:
        phrase: The normalized phrase to match
        template_df: DataFrame containing templates
        threshold: Maximum allowed edit distance ratio (0-1, lower is better)
        ambiguity_threshold: Minimum difference between top matches to avoid ambiguity
        
    Returns:
        tuple: (matching_template_row, index) or raises exception if no good match
    """
    if len(template_df) == 0:
        raise ValueError("Template DataFrame is empty")
    
    # Print debug info
    print(f"Finding match for phrase: '{phrase}'")
    print(f"Template DataFrame has {len(template_df)} rows")
    
    # First try an exact match
    for idx, template_row in template_df.iterrows():
        try:
            template_phrase = extract_template_phrase(template_row)
            if phrase == template_phrase:
                print(f"Found exact match at index {idx}")
                return template_row, idx
        except Exception as e:
            print(f"Error processing template row {idx}: {str(e)}")
    
    # If no exact match, use fuzzy matching with edit distance
    distances = []
    for idx, template_row in template_df.iterrows():
        try:
            template_phrase = extract_template_phrase(template_row)
            # Normalize by max length to get a ratio between 0-1
            max_len = max(len(phrase), len(template_phrase))
            if max_len == 0:  # Handle empty strings
                distance_ratio = 0 if len(template_phrase) == 0 else 1
            else:
                distance = Levenshtein.distance(phrase, template_phrase)
                distance_ratio = distance / max_len
            
            distances.append((distance_ratio, idx, template_row))
            
            # Print debug info for the first few and last few rows
            if idx < 5 or idx >= len(template_df) - 5:
                print(f"  Row {idx}: '{template_phrase}', distance: {distance_ratio:.4f}")
        except Exception as e:
            print(f"Error computing distance for row {idx}: {str(e)}")
    
    # Sort by distance (ascending)
    if not distances:
        raise ValueError(f"No valid template phrases found for comparison")
    
    distances.sort()
    
    # Print the top 3 matches
    print("Top matches:")
    for i in range(min(3, len(distances))):
        dist, idx, row = distances[i]
        phrase_for_display = extract_template_phrase(row)
        print(f"  {i+1}. Index {idx}, distance: {dist:.4f}, phrase: '{phrase_for_display}'")
    
    # Check if we have at least one match within threshold
    if distances[0][0] > threshold:
        raise ValueError(f"No match found for phrase '{phrase}' within threshold {threshold}")
    
    # Check for ambiguity - if the top two matches are too close
    if len(distances) > 1:
        diff = distances[1][0] - distances[0][0]
        if diff < ambiguity_threshold / 100:  # Convert percentage to ratio
            raise ValueError(f"Ambiguous match for phrase '{phrase}'. Top matches have similar distances: {distances[0][0]:.4f} vs {distances[1][0]:.4f}")
    
    # Return the best match
    return distances[0][2], distances[0][1]

def process_csv_file(input_file, template_df, output_folder, match_threshold=0.2, ambiguity_threshold=2):
    """
    Process a single CSV file and create a completed version
    
    Args:
        input_file: Path to input CSV file
        template_df: DataFrame containing templates
        output_folder: Folder to save the completed file
        match_threshold: Maximum allowed edit distance ratio (0-1)
        ambiguity_threshold: Percentage difference required between top matches
    """
    print(f"\nProcessing file: {input_file}")
    
    # Read the input CSV
    input_df = pd.read_csv(input_file)
    print(f"Input CSV has {len(input_df)} rows and columns: {input_df.columns.tolist()}")
    
    # Print a sample row for debugging
    if len(input_df) > 0:
        print("Sample row from input:")
        print(input_df.iloc[0].to_dict())
    
    # Create columns for new attributes
    # Get all column names except the first one which is the template phrase
    if len(template_df.columns) > 1:
        attribute_cols = template_df.columns[1:].tolist()
    else:
        # If template has only one column, check if we can get attribute columns by numbers
        attribute_cols = []
        for col_idx in range(1, min(7, len(template_df.columns) if hasattr(template_df, 'columns') else 0)):
            if col_idx in template_df.columns:
                attribute_cols.append(col_idx)
    
    print(f"Attribute columns from template: {attribute_cols}")
    
    # Add new columns to the input dataframe
    for attr in attribute_cols:
        input_df[attr] = None
    
    # Add column for template index
    input_df['index'] = -1
    input_df['template_index'] = -1
    input_df['Phrase_type'] = -1  # Ensure we always have this column
    
    # List to track errors
    errors = []
    
    # Process each row
    for idx, row in input_df.iterrows():
        try:
            phrase = extract_main_phrase(row)
            print(f"\nProcessing row {idx}, phrase: '{phrase}'")
            
            matching_template, template_idx = find_best_match(
                phrase, 
                template_df, 
                threshold=match_threshold,
                ambiguity_threshold=ambiguity_threshold
            )
            
            print(f"Best match found at index {template_idx}")
            
            # Add template index
            input_df.at[idx, 'template_index'] = template_idx
            input_df.at[idx, 'index'] = idx
            input_df.at[idx, 'Sentence_Stimuli'] = template_df['Sentence_Stimuli'][idx]
            
            # Always set Phrase_type to the template index
            input_df.at[idx, 'Phrase_type'] = template_idx
            
            # Set phrase_type to the template index (if it exists with a different case)
            phrase_type_col = next((col for col in attribute_cols if str(col).lower() == 'phrase_type'), None)
            if phrase_type_col and phrase_type_col != 'Phrase_type':
                input_df.at[idx, phrase_type_col] = template_idx
            
            # Add all attributes from template
            for attr in attribute_cols:
                try:
                    input_df.at[idx, attr] = matching_template[attr]
                except KeyError:
                    # If attribute doesn't exist in the template, skip it
                    print(f"  Warning: Attribute '{attr}' not found in template")
                
        except ValueError as e:
            # Track error with row information
            error_msg = f"Row {idx}: {str(e)}"
            print(f"  Error: {error_msg}")
            errors.append(error_msg)
    
    # If we have any errors, raise an exception
    if errors:
        error_message = f"Failed to process {input_file} due to the following errors:\n" + "\n".join(errors)
        raise ValueError(error_message)
    
    # Get a list of all column names
    all_columns = input_df.columns.tolist()
    all_columns.remove('index')
    all_columns.remove('template_index')
    all_columns.remove('Sentence_Stimuli')
    new_column_order = ['index', 'template_index','Sentence_Stimuli'] + all_columns
    input_df = input_df[new_column_order]

    # Create output filename
    base_name = os.path.basename(input_file)
    file_name, file_ext = os.path.splitext(base_name)
    output_file = os.path.join(output_folder, f"{file_name}_complete{file_ext}")
    
    # Save to output file
    input_df.to_csv(output_file, index=False)
    print(f"Processed {input_file} -> {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Complete CSV files with template attributes')
    parser.add_argument('template_path', help='Path to the template XLSX file')
    parser.add_argument('input_folder', help='Path to folder containing input CSV files')
    parser.add_argument('output_folder', help='Path to folder for saving output CSV files')
    parser.add_argument('--threshold', type=float, default=0.2, 
                        help='Fuzzy matching threshold (0-1, lower is better)')
    parser.add_argument('--ambiguity', type=float, default=2.0,
                        help='Ambiguity threshold percentage between top matches')
    parser.add_argument('--sheet', type=str, default=None,
                        help='Sheet name in the Excel template file')
    parser.add_argument('--header-row', type=int, default=0,
                        help='Row index to use as header (default: 0)')
    parser.add_argument('--skip-rows', type=int, default=None,
                        help='Number of rows to skip at the beginning of the Excel file')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Read the template XLSX
    print(f"Loading template from: {args.template_path}")
    try:
        # Get the sheet name to use
        sheet_to_use = args.sheet
        if sheet_to_use is None:
            # If no sheet specified, get the first sheet name
            xls = pd.ExcelFile(args.template_path)
            sheet_to_use = xls.sheet_names[0]
            print(f"Using first sheet: {sheet_to_use}")
        
        # Read all sheets to handle different Excel structures
        # Try both with and without a header
        try:
            # First attempt: Read with header at the specified row
            template_df = pd.read_excel(
                args.template_path, 
                sheet_name=sheet_to_use,
                header=args.header_row,
                skiprows=args.skip_rows
            )
            print(f"Successfully loaded template sheet '{sheet_to_use}' with {len(template_df)} rows using header")
        except Exception as e1:
            print(f"Error reading with header: {str(e1)}")
            try:
                # Second attempt: Read without a header
                template_df = pd.read_excel(
                    args.template_path, 
                    sheet_name=sheet_to_use,
                    header=None,
                    skiprows=args.skip_rows
                )
                print(f"Successfully loaded template sheet '{sheet_to_use}' with {len(template_df)} rows without header")
            except Exception as e2:
                print(f"Error reading without header: {str(e2)}")
                print("Available sheets:")
                xls = pd.ExcelFile(args.template_path)
                for sheet in xls.sheet_names:
                    print(f"  - {sheet}")
                return
        
        # Display template info for debugging
        print(f"Template columns: {template_df.columns.tolist()}")
        if len(template_df) > 0:
            print("First row sample data:")
            print(template_df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error loading template file: {str(e)}")
        print("Available sheets:")
        try:
            xls = pd.ExcelFile(args.template_path)
            for sheet in xls.sheet_names:
                print(f"  - {sheet}")
        except Exception as e2:
            print(f"Could not list sheets: {str(e2)}")
        return
    
    # Process each CSV file in the input folder
    successful = 0
    failed = 0
    error_messages = []
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(args.input_folder) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process files with progress bar
    for file_name in tqdm(csv_files, desc="Processing files"):
        input_file = os.path.join(args.input_folder, file_name)
        try:
            process_csv_file(
                input_file, 
                template_df, 
                args.output_folder,
                match_threshold=args.threshold,
                ambiguity_threshold=args.ambiguity
            )
            successful += 1
            print(f"✓ Successfully processed: {file_name}")
        except Exception as e:
            failed += 1
            error_message = f"✗ Failed to process {file_name}: {str(e)}"
            print(error_message)
            error_messages.append(error_message)
    
    # Print summary
    print("\n" + "="*80)
    print(f"Processing complete. {successful} files processed successfully, {failed} files failed.")
    
    # If there were failures, provide error details
    if error_messages:
        print("\nError details:")
        for msg in error_messages:
            print(f"- {msg}")

if __name__ == "__main__":
    main()