"""
Small file with function that gets stimuli phrases from an excel file. Should barely exist as a separate file.
"""

import pandas as pd

def extract_phrases(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Filter rows where phrase_type column equals 0
    # filtered_df = df[df['Phrase_type'] == 0]
    filtered_df = df
    
    # Extract the Sentence_Stimuli column values
    phrases = filtered_df['Sentence_Stimuli'].tolist()
    
    return phrases
