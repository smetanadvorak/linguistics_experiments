import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple
import copy
import json
import math

dynamic_aoi = {}

class XMLProcessor:
    namespace = {
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsd': 'http://www.w3.org/2001/XMLSchema'
        }

    def __init__(self, input_file: str):
        self.input_file = input_file

    @staticmethod
    def advance_aoi_time(aoi, t0):
        ''' Moves all of dynamic aoi's timestamps so that it starts at t '''
        keyframes = aoi.find('KeyFrames').findall('KeyFrame')
        timestamps = [int(kf.find('Timestamp').text) for kf in keyframes]
        timestamps = [t - min(timestamps) for t in timestamps]
        timestamps = [t + int(t0 * 1e6) for t in timestamps]
        for kf, ts in zip(keyframes, timestamps):
            ts_field = kf.find('Timestamp')
            ts_field.text = str(ts)
        return aoi
    
    @staticmethod
    def set_aoi_start_stop(aoi, t0, t1):
        keyframes = aoi.find('KeyFrames').findall('KeyFrame')
        assert(len(keyframes) == 2)
        assert(keyframes[0].find('Visible').text == 'true')
        assert(keyframes[1].find('Visible').text == 'false')
        # Set the first keyframe to t0
        # ts0 = int(math.floor(t0 * 10) * 1e5)
        # ts1 = int(math.floor(t1 * 10) * 1e5)
        ts0 = int(t0 * 1e6)
        ts1 = int(t1 * 1e6)
        keyframes[0].find('Timestamp').text = str(ts0)
        keyframes[1].find('Timestamp').text = str(ts1)
        return aoi
    
    @staticmethod
    def shift_aoi(aoi, new_origin, old_origin):
        ''' Moves all of dynamic aoi's timestamps so that it starts at t '''
        keyframes = aoi.find('KeyFrames').findall('KeyFrame')
        for kf in keyframes:
            points_list = kf.find('Points').findall('Point')
            for point in points_list:
                point.find('X').text = str(int(point.find('X').text) - old_origin[0] + new_origin[0])
                point.find('Y').text = str(int(point.find('Y').text) - old_origin[1] + new_origin[1])
        return aoi

    def read_xml(self, drop_first_chars=1):
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
            item_id = item_name[drop_first_chars:]
            self.master_dict[item_name] = copy.deepcopy(aoi)

    def generate_experiment(self, event_list):
        output_list = []
        n_events_accepted = 0
        template_events_left = set(list(self.master_dict.keys()))
        print(template_events_left)
        experiment_events_left = set([e['event'] for e in event_list])
        for event in event_list:
            item_name = event['event']
            item_t0 = event['start_time'] - 0.2
            item_name = item_name[:4]  # to exclude the last '4' character from all item names (present in csv)
            if not item_name in self.master_dict:
                print(f"AOI for event {item_name} not found in AOI template!")
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


    @staticmethod
    def write_xml(aoi_list: List[Tuple[int, ET.Element]], output_file: str):
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

        ET.register_namespace('xsi', XMLProcessor.namespace['xsi'])
        ET.register_namespace('xsd', XMLProcessor.namespace['xsd'])

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