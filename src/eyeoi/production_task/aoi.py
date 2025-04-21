import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple
import copy
import json

known_prefixes = {'G', 'H', 'B'}

dynamic_aoi = {}

# phrase_color_mapping = {
#     'D1': 'NamedColor:Silver',
#     'D2': "NamedColor:Gainsboro",
#     'TC': "NamedColor:MediumPurple",
#     'PS': "NamedColor:DimGray",
#     'SR': "NamedColor:DarkGray",
#     'TX': "NamedColor:LavenderBlush"}

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

    def advance_aoi_time(self, aoi, t0):
        ''' Moves all of dynamic aoi's timestamps so that it starts at t '''
        keyframes = aoi.find('KeyFrames').findall('KeyFrame')
        timestamps = [int(kf.find('Timestamp').text) for kf in keyframes]
        timestamps = [t - min(timestamps) for t in timestamps]
        timestamps = [t + int(t0 * 1e6) for t in timestamps]
        for kf, ts in zip(keyframes, timestamps):
            ts_field = kf.find('Timestamp')
            ts_field.text = str(ts)
        return aoi

    def read_xml(self):
        """
        Read XML file and return a list of tuples containing (id, element) pairs.
        """
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(self.input_file, parser=parser)
        root = tree.getroot()

        self.original_root = root
        self.tree = tree
        self.master_dict = {}

        for aoi in root.findall('DynamicAOI'):
            item_name = copy.copy(aoi.find('Name').text)
            item_id = item_name[1:]

            if not item_id in self.master_dict:
                self.master_dict[item_id] = {}
            self.master_dict[item_id][item_name] = copy.deepcopy(aoi)

    def generate_experiment(self, event_list):
        output_list = []
        n_events_accepted = 0
        template_events_left = set(list(self.master_dict.keys()))
        print(template_events_left)
        experiment_events_left = set([e['event'] for e in event_list])
        for event in event_list:
            item_name = event['event']
            item_t0 = event['start_time'] - 0.1
            item_name = item_name[:4]  # to exclude the last '4' character from all item names (present in csv)
            if not item_name in self.master_dict:
                # print(f"AOI for event {item_name} not found in AOI template!")
                continue

            n_events_accepted += 1
            experiment_events_left.remove(event['event'])
            template_events_left.remove(item_name)
            for aoi in self.master_dict[item_name].values():
                modified_aoi = self.advance_aoi_time(copy.deepcopy(aoi), item_t0)
                output_list.append(modified_aoi)

        if n_events_accepted != len(self.master_dict):
            msg = f"Got AOIs for {n_events_accepted} items instead of all {len(self.master_dict)} items present in template AOI. Missing: {template_events_left}"
            print(msg)
            input("Continue ...")
        if len(experiment_events_left):
            msg = f"Got AOIs for {n_events_accepted} items instead of all {len(event_list)} items present in the experiment. Missing: {experiment_events_left}"
            print(msg)
            # input("Continue ...")
            
            # raise Exception(msg)
        return output_list


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