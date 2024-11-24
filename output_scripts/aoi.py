import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple
import copy
import json

def extract_suffix(text):
    known_prefixes = {'NP', 'ADJ', 'INF', 'SUB', 'OBJ'}
    
    # Check each known prefix
    for prefix in known_prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    
    # If no known prefix is found, return None or raise an exception
    return None  # or raise ValueError(f"No known prefix found in {text}")

def swap_keyframes(aoi1: ET.Element, aoi2: ET.Element):
    """
    Swap the KeyFrames elements between two AOI XML elements.
    
    Args:
        aoi1: First DynamicAOI XML element
        aoi2: Second DynamicAOI XML element
    """
    # Get KeyFrames elements
    keyframes1 = aoi1.find('KeyFrames')
    keyframes2 = aoi2.find('KeyFrames')
    
    if keyframes1 is None or keyframes2 is None:
        raise ValueError("One or both AOIs are missing KeyFrames element")
    
    # Create deep copies of the KeyFrames elements
    keyframes1_copy = copy.deepcopy(keyframes1)
    keyframes2_copy = copy.deepcopy(keyframes2)
    
    # Remove original KeyFrames elements
    aoi1.remove(keyframes1)
    aoi2.remove(keyframes2)
    
    # Add the swapped KeyFrames elements
    aoi1.append(keyframes2_copy)
    aoi2.append(keyframes1_copy)

def replace_suffix_in_name(aoi_element: ET.Element, start_time, stop_time, new_suffix: str) -> ET.Element:
    """
    Replace the suffix in the Name field of an AOI element while preserving the prefix.
    
    Args:
        aoi_element: Single AOI XML element
        new_suffix: The new suffix to use as replacement
        
    Returns:
        Modified AOI XML element
    """
    # Known prefixes from the groups
    known_prefixes = {'NP', 'ADJ', 'INF', 'SUB', 'OBJ'}
    
    # Make a deep copy to avoid modifying the original
    aoi_copy = copy.deepcopy(aoi_element)
    
    # Find the Name element
    name_elem = aoi_copy.find('Name')
    if name_elem is not None:
        current_name = name_elem.text
        # Find the prefix by looking for known group names
        for prefix in known_prefixes:
            if current_name.startswith(prefix):
                # Replace everything after the prefix with new suffix
                new_name = prefix + new_suffix
                name_elem.text = new_name
                break

    kfs_elem = aoi_copy.find('KeyFrames')
    kfs = kfs_elem.findall('KeyFrame')
    # print(kfs[0].find('Timestamp').text, kfs[1].find('Timestamp').text)
    kfs[0].find('Timestamp').text = str(int(start_time * 1e6))
    kfs[1].find('Timestamp').text = str(int(stop_time * 1e6))
    # print(kfs[0].find('Timestamp').text, kfs[1].find('Timestamp').text)
    return aoi_copy



class XMLProcessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.namespace = {
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsd': 'http://www.w3.org/2001/XMLSchema'
        }
        
        # Simple mapping between sentences and their base words
        self.sentence_mapping = {
            "This glass is easy to drop.": "glass",
            "This game is easy to play.": "game",
            "This girl is difficult to see.": "girl",
            "This cat is easy to wash.": "cat",
            "This bird is easy to chase.": "bird",
            "This picture is easy to draw.": "pic",
            "This ball is difficult to bounce.": "ball",
            "This patient is difficult to move.": "pat",
            "This dog is difficult to walk.": "dog",
            "This trolley is difficult to roll.": "trol",
            "This animal is easy to hide.": "anim",
            "This book is difficult to read.": "book"
        }

        self.sentence_mapping = {' '.join(k.split()): v for k,v in self.sentence_mapping.items()}
        
    def read_xml(self):
        """
        Read XML file and return a list of tuples containing (id, element) pairs.
        """
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(self.input_file, parser=parser)
        root = tree.getroot()
        
        self.original_root = root
        self.tree = tree
        
        aoi_list = []

        for aoi in root.findall('DynamicAOI'):
            aoi_list.append(copy.deepcopy(aoi))

        item_dict = {}
        n_items = 12
        for item_idx in range(n_items):
            item_aoi_list = aoi_list[item_idx * 5 : (item_idx + 1) * 5]
            name_elem = item_aoi_list[0].find('Name')
            name = extract_suffix(name_elem.text)
            print(item_idx, name)
            item_dict[name] = item_aoi_list

        return aoi_list[::-1], item_dict

    def reorder_by_sentences(self, aoi_list, sentences_df, t_shift, events_dict):
        
        flipped_origin = [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]

        first_row_idx = 5
        t0 = -t_shift + float(sentences_df.iloc[first_row_idx]['Stimuli_text.started'])

        sentence_idx = 0
        for item_idx in range(first_row_idx, first_row_idx + 18):
            sentence = sentences_df.iloc[item_idx]['Sentences']
            is_stimuli = int(sentences_df.iloc[item_idx]['Type_stimuli'])
            is_flipped = int(sentences_df.iloc[item_idx]['flipped'])
            is_flipped = is_flipped != flipped_origin[sentence_idx]

            if not is_stimuli:
                print("Not stimuli")
                continue

            if pd.isna(sentence):
                print("None sentence")
                exit()
                
            sentence = ' '.join(sentence.split())

            if sentence not in self.sentence_mapping:
                print("Unknown sentence:", sentence)
                exit()

            item = self.sentence_mapping[sentence]
            print(f"Sentence: {sentence}, item: {item}")

            # start_time = float(sentences_df.iloc[item_idx]['Stimuli_text.started']) - t0
            # stop_time = start_time + float(sentences_df.iloc[item_idx]['mouse_2.time'])
            event = events_dict[item]
            start_time = float(event["start_time"]) - 0.1
            stop_time = float(event["end_time"]) - 0.1

            sub_idx = None
            obj_idx = None
            for aoi_idx in range(sentence_idx * 5, (sentence_idx + 1) * 5):
                name_elem = aoi_list[aoi_idx].find('Name')
                current_name = name_elem.text
                print(aoi_idx, current_name, start_time, stop_time)
                aoi_list[aoi_idx] = replace_suffix_in_name(aoi_list[aoi_idx], start_time, stop_time, item)
                if "OBJ" in current_name:
                    obj_idx = aoi_idx
                if "SUB" in current_name:
                    sub_idx = aoi_idx                    

            # swap obj and sub keyframes
            if is_flipped:
                swap_keyframes(aoi_list[obj_idx], aoi_list[sub_idx])

            sentence_idx += 1
        
        return aoi_list
    
    def reorder_by_sentences2(self, item_dict, sentences_df, events_dict):
        
        flipped_origin = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        flipped_origin_dict = {
            "glass":    1,
            "game":     1,
            "girl":     0,
            "cat":      0,
            "bird":     0,
            "pic":      1,
            "ball":     0,
            "pat":      1,
            "dog":      1,
            "trol":     1,
            "anim":     1,
            "book":     0
        }

        first_row_idx = 5

        sentence_idx = -1
        output_list = []
        for row_idx in range(first_row_idx, first_row_idx + 18):
            sentence_idx += 1
            is_stimuli = int(sentences_df.iloc[row_idx]['Type_stimuli'])
            is_flipped = int(sentences_df.iloc[row_idx]['flipped'])

            if not is_stimuli:
                print("Not stimuli")
                continue
                
            sentence = sentences_df.iloc[row_idx]['Sentences']
            sentence = ' '.join(sentence.split())

            if sentence not in self.sentence_mapping:
                print("Unknown sentence:", sentence)
                exit()

            if pd.isna(sentence):
                print("None sentence")
                exit()

            item_name = self.sentence_mapping[sentence]
            is_flipped = (is_flipped != flipped_origin_dict[item_name])
            
            print(f"Sentence: {sentence}, item: {item_name}")

            # start_time = float(sentences_df.iloc[item_idx]['Stimuli_text.started']) - t0
            # stop_time = start_time + float(sentences_df.iloc[item_idx]['mouse_2.time'])
            event = events_dict[item_name]
            start_time = float(event["start_time"]) - 0.1
            stop_time = float(event["end_time"]) - 0.1
            print(item_name, start_time, stop_time)

            sub_idx = None
            obj_idx = None
            for aoi_idx in range(5):
                aoi = replace_suffix_in_name(item_dict[item_name][aoi_idx], start_time, stop_time, item_name)
                name_elem = item_dict[item_name][aoi_idx].find('Name')
                current_name = name_elem.text
                if "OBJ" in current_name:
                    obj_idx = len(output_list)
                if "SUB" in current_name:
                    sub_idx = len(output_list)
                output_list.append(aoi)

            # swap obj and sub keyframes
            if is_flipped:
                swap_keyframes(output_list[obj_idx], output_list[sub_idx])

        

        chunks = [output_list[i:i + 5] for i in range(0, len(output_list), 5)]
        chunks.reverse()
        output_list = [num for chunk in chunks for num in chunk]
        
        # output_list = output_list[::-1]
        for idx, aoi in enumerate(output_list):
            id_elem = aoi.find('ID')
            id_elem.text = str(idx)

        return output_list[::-1]
    
    def write_xml(self, aoi_list: List[Tuple[int, ET.Element]], output_file: str):
        """
        Write the reordered AOIs to a new XML file.
        """
        # new_root = ET.Element('ArrayOfDynamicAOI', attrib=self.original_root.attrib)
        new_root = ET.Element('ArrayOfDynamicAOI', {
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xmlns:xsd': 'http://www.w3.org/2001/XMLSchema'
        })
        
        # Sort by ID before writing
        for aoi in aoi_list[::-1]:
            new_root.append(copy.deepcopy(aoi))
        
        tree = ET.ElementTree(new_root)
        
        ET.register_namespace('xsi', self.namespace['xsi'])
        ET.register_namespace('xsd', self.namespace['xsd'])
        
        with open(output_file, 'wb') as f:
            f.write(b'<?xml version="1.0"?>\n')
            tree.write(f, encoding='utf-8', xml_declaration=False, method='xml')
        print(f"\nWritten reordered XML to {output_file}")

def main():
    sentences_df = pd.read_csv('G20407.csv')
    processor = XMLProcessor('G20406.xml') # do not change this

    t0 = 30.641

    events_file = "detected_events.json"
    with open(events_file, 'r') as f:
        events_data = json.load(f)
        events_dict = {event['event']: event for event in events_data}

    print("\nReading XML file...")
    aoi_list, item_dict = processor.read_xml()
    print(len(aoi_list))
    
    print("\nReordering entries based on sentences...")
    # ordered_aoi_list = processor.reorder_by_sentences(aoi_list, sentences_df, t0, events_dict)
    ordered_aoi_list = processor.reorder_by_sentences2(item_dict, sentences_df, events_dict)

    processor.write_xml(ordered_aoi_list, 'output31.xml')

if __name__ == "__main__":
    main()