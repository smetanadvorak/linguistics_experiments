import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple
import copy
import json

from .text_matching_experiment import sentence_mapping

known_prefixes = {'D1', 'D2', 'TC', 'PS', 'SR', 'TX'}

phrase_type_mapping = {0: 'TC', 1: 'SR', 2: 'D1', 3: 'D2', 4: 'PS'}

phrase_color_mapping = {
    'D1': 'NamedColor:Silver',
    'D2': "NamedColor:Gainsboro",
    'TC': "NamedColor:MediumPurple",
    'PS': "NamedColor:DimGray",
    'SR': "NamedColor:DarkGray",
    'TX': "NamedColor:LavenderBlush"}

def extract_prefix_suffix(text):
    # Check each known prefix
    for prefix in known_prefixes:
        if text.startswith(prefix):
            return prefix, text[len(prefix):]

    # If no known prefix is found, return None or raise an exception
    return None  # or raise ValueError(f"No known prefix found in {text}")


def make_aoi(aoi_element: ET.Element, start_time, stop_time, name: str) -> ET.Element:
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
        name_elem.text = name
        # Find the prefix by looking for known group names
        for prefix in known_prefixes:
            if name.startswith(prefix):
                color_elem = aoi_copy.find('Color')
                color_elem.text = phrase_color_mapping[prefix]
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
        self.sentence_mapping = sentence_mapping

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


        box_list = []
        for box_idx in range(6):
            # name_elem = aoi_list[box_idx].find('Name')
            # box_type, item_name = extract_prefix_suffix(name_elem.text)
            box_list.append(aoi_list[box_idx])

        height_list = []
        for box in box_list:
            height = int(box.find('KeyFrames').findall('KeyFrame')[0].find('Points').findall('Point')[0].find('Y').text)
            height_list.append(height)

        sorted_pairs = sorted(zip(height_list, box_list), key=lambda x : x[0])
        box_list = [x[1] for x in sorted_pairs]
        box_dict = {'TX': box_list[0]}
        for idx, letter in zip(range(1, 6), ['a', 'b', 'c', 'd', 'e']):
            box_dict[letter] = box_list[idx]

        # [print(k, box.find('KeyFrames').findall('KeyFrame')[0].find('Points').findall('Point')[0].find('Y').text) for k, box in box_dict.items()]

        # box_dict_type = {}
        # n_items = 16
        # for item_idx in range(n_items):
        #     item_aoi_list = aoi_list[item_idx * 6 : (item_idx + 1) * 6]
        #     name_elem = item_aoi_list[0].find('Name')
        #     name = extract_prefix_suffix(name_elem.text)
        #     box_dict_type[name] = item_aoi_list

        return box_dict

    def reorder_by_sentences(self, box_dict, sentences_df, events_dict):
        first_row_idx = 0

        sentence_idx = -1
        output_list = []
        for row_idx in range(first_row_idx, first_row_idx + 40):
            sentence_idx += 1

            sentence = sentences_df.iloc[row_idx]['Text']
            sentence = ' '.join(sentence.split())
            print(sentence)

            is_stimuli = False
            for s in self.sentence_mapping:
                if sentence.startswith(s):
                    is_stimuli = True
                    item_name = self.sentence_mapping[s]

            if not is_stimuli:
                print("Not stimuli, skipping")
                continue

            if pd.isna(sentence):
                print("None sentence")
                continue

            box_order = []
            for box_idx in range(5):
                column = f"Phrase_type{box_idx+1}"
                phrase_type = sentences_df.iloc[row_idx][column]
                phrase_type = phrase_type_mapping[int(phrase_type)]
                box_order.append(phrase_type)

            print(f"Sentence: {sentence}, item: {item_name}")

            event = events_dict[item_name]
            start_time = round(float(event["start_time"]), 1)
            stop_time = round(float(event["end_time"]), 1)
            print(item_name, start_time, stop_time)

            # make TX box
            aoi = make_aoi(box_dict['TX'], start_time, stop_time, 'TX'+item_name)
            output_list.append(aoi)

            for aoi_idx, aoi_type in zip(['a', 'b', 'c', 'd', 'e'], box_order):
                aoi = make_aoi(box_dict[aoi_idx], start_time, stop_time, aoi_type+item_name)
                output_list.append(aoi)

        chunks = [output_list[i:i + 6] for i in range(0, len(output_list), 6)]
        chunks.reverse()
        output_list = [num for chunk in chunks for num in chunk]

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
    template_xml = XMLProcessor('G20406.xml') # do not change this

    events_file = "detected_events.json"
    with open(events_file, 'r') as f:
        events_data = json.load(f)
        events_dict = {event['event']: event for event in events_data}

    print("\nReading XML file...")
    box_dict = template_xml.read_xml()
    # print(len(aoi_list))

    print("\nReordering entries based on sentences...")
    ordered_aoi_list = template_xml.reorder_by_sentences(box_dict, sentences_df, events_dict, flipped_origin_dict_en)

    template_xml.write_xml(ordered_aoi_list, 'output31.xml')

if __name__ == "__main__":
    main()