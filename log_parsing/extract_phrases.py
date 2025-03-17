import pandas as pd
import sys

def extract_phrases(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Filter rows where phrase_type column equals 0
    # filtered_df = df[df['Phrase_type'] == 0]
    filtered_df = df
    
    # Extract the Sentence_Stimuli column values
    phrases = filtered_df['Sentence_Stimuli'].tolist()
    
    return phrases

def main():
    file_path = sys.argv[1]  # put with your actual file path
    
    try:
        phrases = extract_phrases(file_path)
        
        print("Phrases with phrase_type 0:")
        for i, phrase in enumerate(phrases, 1):
            print(f"{i}. {phrase}")
            
        print(f"\nTotal phrases found: {len(phrases)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()