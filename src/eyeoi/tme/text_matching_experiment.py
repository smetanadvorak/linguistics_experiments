import pandas as pd
from typing import List, Dict, Any

tm_columns = ["Text", "order"] + ["Phrase_type" + str(i) for i in range(1,6)]

sentence_mapping = {
    "Unlike movies, where intruders": "dog",
    "I wake up early": "hab",
    "We decided on vinyl": "flo",
    "Some insects use camouflage": "ins",
    "She went on saying": "kid",
    "Generally, Vick": "vic",
    "British people are more": "arr",
    "A Martinsburg orchard lost": "bug",
    "There are many small": "txt",
    "The Nigerian President speaks": "pre",
    "To a child, moving": "chi",
    "Most of these roses": "flw",
    "In fact, you can": "tan",
    "Despite their large size": "ele",
    "All types of soft": "str",
    "I also appreciated that": "mus",

    "Contrairement aux films, où les intrus": "dog",
    "Je me réveille tôt": "hab",
    "Nous avons opté pour un revêtement de sol en vinyle": "flo",
    "Certains insectes utilisent le camouflage": "ins",
    "Elle a poursuivi en disant": "kid",
    "En général, la voix de Vick": "vic",
    "Les Anglais sont plus": "arr",
    "Un verger de Martinsburg a perdu": "bug",
    "Il y a beaucoup de petites": "txt",
    "Le président Nigérian parle": "pre",
    "Pour un enfant, le déménagement": "chi",
    "La plupart de ces rosiers": "flw",
    "Effectivement, on ne peut": "tan",
    "Malgré leur grande taille": "ele",
    "Tous les types de fruits": "str",
    "J'ai également apprécié": "mus"
}


sentence_mapping = {' '.join(k.split()): v for k,v in sentence_mapping.items()}


def from_excel(file_path: str, columns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Load data from an Excel table into a list of dictionaries with specified columns.

    Args:
        file_path (str): Path to the Excel file
        columns (List[str], optional): List of column names to include. If None, all columns are included.

    Returns:
        List[Dict[str, Any]]: List of dictionaries where each dictionary represents a row
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    df = df.head(40)

    # If specific columns are requested, filter to only those columns
    if columns is not None:
        # Verify all requested columns exist
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Requested columns not found in file: {missing_cols}")
        df = df[columns]

    # Convert DataFrame to list of dictionaries
    dict_list = df.to_dict(orient='records')
    for d in dict_list:
        for sentence, item_type in sentence_mapping.items():
            if ' '.join(d['Text'].split()).startswith(sentence):
                d['item_type'] = item_type
        if 'item_type' not in d:
            d['item_type'] = None
    return dict_list

def get_item_order(filepath):
    data = from_excel(filepath)
    data = sorted(data, key=lambda x: int(x['order']))
    item_list = []
    for d in data:
        if d['item_type'] is not None:
            item_list.append(d['item_type'])
    return item_list


def get_item_order_time(filepath):
    data = from_excel(filepath)
    data = sorted(data, key=lambda x: int(x['order']))
    item_list = []
    for d in data:
        if d['item_type'] is not None:
            entry = {'item': d['item_type'], 'start': d['Stimuli_choice.started_raw']}
            entry['stop'] = entry['start'] + d['Stimuli_choice.time_raw']
            item_list.append(entry)
    return item_list


# Example usage:
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    data = from_excel(filepath)

    item_order = get_item_order(filepath)
    print(' '.join(item_order))