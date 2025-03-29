"""AOI rearrangement based on video-extracted events for experiment on 18/03/25"""

import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple
import copy
import json
import glob
import os


known_prefixes = {'INF', 'ADJ', 'NP'}

def extract_suffix(text):
    # Check each known prefix
    for prefix in known_prefixes:
        if text.startswith(prefix):
            return text[len(prefix):], prefix
    
    # If no known prefix is found, return None or raise an exception
    return None  # or raise ValueError(f"No known prefix found in {text}")

# def swap_keyframes(aoi1: ET.Element, aoi2: ET.Element):
#     """
#     Swap the KeyFrames elements between two AOI XML elements.
    
#     Args:
#         aoi1: First DynamicAOI XML element
#         aoi2: Second DynamicAOI XML element
#     """
#     # Get KeyFrames elements
#     keyframes1 = aoi1.find('KeyFrames')
#     keyframes2 = aoi2.find('KeyFrames')
    
#     if keyframes1 is None or keyframes2 is None:
#         raise ValueError("One or both AOIs are missing KeyFrames element")
    
#     # Create deep copies of the KeyFrames elements
#     keyframes1_copy = copy.deepcopy(keyframes1)
#     keyframes2_copy = copy.deepcopy(keyframes2)
    
#     # Remove original KeyFrames elements
#     aoi1.remove(keyframes1)
#     aoi2.remove(keyframes2)
    
#     # Add the swapped KeyFrames elements
#     aoi1.append(keyframes2_copy)
#     aoi2.append(keyframes1_copy)

def replace_suffix_in_name(aoi_element: ET.Element, start_time, stop_time, new_suffix: str) -> ET.Element:
    """
    Replace the suffix in the Name field of an AOI element while preserving the prefix.
    
    Args:
        aoi_element: Single AOI XML element
        new_suffix: The new suffix to use as replacement
        
    Returns:
        Modified AOI XML element
    """
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
        n_items = 16
        for item_idx in range(n_items):
            item_aoi_list = aoi_list[item_idx * len(known_prefixes) : (item_idx + 1) * len(known_prefixes)]
            name_elem = item_aoi_list[0].find('Name')
            name, prefix = extract_suffix(name_elem.text)
            print(item_idx, name)
            item_dict[name] = item_aoi_list

        return aoi_list[::-1], item_dict
    
    def reorder_by_sentences(self, item_dict, events_dict):

        output_list = []
        for item_name, event in events_dict.items():
            print(item_name, event)
            print(item_dict[item_name])
            for aoi_idx in range(len(known_prefixes)):
                aoi = item_dict[item_name][aoi_idx]
                aoi_copy = copy.deepcopy(aoi)
                name_elem = aoi_copy.find('Name')
                kfs_elem = aoi_copy.find('KeyFrames')
                kfs = kfs_elem.findall('KeyFrame')
                kfs[0].find('Timestamp').text = str(int(event['start_time'] * 1e6))
                kfs[1].find('Timestamp').text = str(int(event['end_time'] * 1e6))
                output_list.append(aoi_copy)

        chunks = [output_list[i:i + 3] for i in range(0, len(output_list), 3)]
        chunks.reverse()
        output_list = [num for chunk in chunks for num in chunk]
        
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



def process_event_file(events_path, output_path):
    template_xml = XMLProcessor('data/aoi2/fr/G10106 FR (AOIs) manual.xml')

    with open(events_path, 'r') as f:
        events_data = json.load(f)
        events_dict = {event['event'][:3]: event for event in events_data}

    print("\nReading XML file...")
    aoi_list, item_dict = template_xml.read_xml()
    print("Total aoi:", len(aoi_list), ", items: ", len(item_dict))
    
    print("\nReordering entries based on sentences...")
    ordered_aoi_list = template_xml.reorder_by_sentences(item_dict, events_dict)

    template_xml.write_xml(ordered_aoi_list, output_path)

if __name__ == "__main__":
    folder = "data/aoi2/fr/events"
    event_paths = glob.glob(os.path.join(folder, '*.json'))
    out_dir = "data/aoi2/fr/output_aoi"
    os.makedirs(out_dir, exist_ok=True)

    for event_path in event_paths:
        out_name = os.path.basename(event_path).split('.')[0][:-7] + ".xml"
        print(out_name)
        out_path = os.path.join(out_dir, out_name)
        process_event_file(event_path, out_path)
