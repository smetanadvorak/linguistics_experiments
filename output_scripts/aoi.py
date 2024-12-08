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
            # English mappings
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
            "This book is difficult to read.": "book",
            
            # French mappings
            "Ce verre est facile à renverser.": "glass",
            "Ce jeu est facile à jouer.": "game",
            "Cette fille est difficile à voir.": "girl",
            "Ce chat est facile à laver.": "cat",
            "Cet oiseau est facile à chasser.": "bird",
            "Cette image est facile à dessiner.": "pic",
            "Cette balle est difficile à rebondir.": "ball",
            "Ce ballon est difficile à frapper.": "ball",
            "Ce patient est difficile à déplacer.": "pat",
            "Ce chien est difficile à promener.": "dog",
            "Ce chariot est difficile à pousser.": "trol",
            "Cet animal est facile à cacher.": "anim",
            "Ce livre est difficile à lire.": "book",
            
            # Russian mappings
            "Этот стакан легко уронить.": "glass",
            "В эту игру легко играть.": "game",
            "В эту игру сложно играть.": "game",
            "Эту девочку сложно видеть.": "girl",
            "Эту кошку легко мыть.": "cat",
            "Эту птицу легко догонять.": "bird",
            "Эту картинку легко рисовать.": "pic",
            "Этот мяч сложно отбивать.": "ball",
            "Этого пациента сложно передвигать.": "pat",
            "Эту собаку сложно выгуливать.": "dog",
            "Эту тележку сложно толкать.": "trol",
            "Это животное легко прятать.": "anim",
            "Эту книгу сложно читать.": "book",
            
            "Эту тележку сложно катить.": "trol",
            "Эту девочку сложно видеть.": "girl",
            "Этот стакан легко уронить.": "glass",
            "Эту картинку легко рисовать.": "pic",
            "Стакан пустой.": "glass",
            "Эту книгу сложно читать.": "book",
            "Это животное легко прятать.": "anim",
            "Окно открыто.": "wind",
            "Курица гонится за мальчиком.": "chick",
            "Эту кошку легко мыть.": "cat",
            "Птица сидит на носороге.": "bird",
            "В эту игру сложно играть.": "game",
            "Эту собаку сложно выгуливать.": "dog",
            "Бутылка стоит перед гитарой.": "bottl",
            "Этот мяч сложно вести.": "ball",
            "Эту птицу легко ловить.": "bird",
            "Этого пациента сложно передвигать.": "pat",
            "Собака смотрит на кролика.": "dog"
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
    
    def reorder_by_sentences(self, item_dict, sentences_df, events_dict, flipped_origin_dict):
        first_row_idx = 5

        sentence_idx = -1
        output_list = []
        for row_idx in range(first_row_idx, first_row_idx + 18):
            sentence_idx += 1
            is_stimuli = int(sentences_df.iloc[row_idx]['Type_stimuli'])
            is_flipped = int(sentences_df.iloc[row_idx]['flipped'])

            if not is_stimuli:
                print("Not stimuli, skipping")
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

            event = events_dict.get(item_name, events_dict.get(item_name + "_text.jpg"))
            start_time = round(float(event["start_time"]) - 0.1, 1)
            stop_time = round(float(event["end_time"]) - 0.1, 1)
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


flipped_origin_dict_en = {
    "glass": 0, "game": 1, "girl": 1, "cat": 1, "bird": 0, "pic": 0,
    "ball": 0, "pat": 1, "dog": 0, "trol": 0, "anim": 1, "book": 1
}

flipped_origin_dict_ru = {
    "glass": 0, "game": 1, "girl": 0, "cat": 0, "bird": 0, "pic": 0, 
    "ball": 0, "pat": 1, "dog": 0, "trol": 1, "anim": 1, "book": 0
}

flipped_origin_dict_fr = {
    "glass": 0, "game": 0, "girl": 1, "cat": 0, "bird": 0, "pic": 0,
    "ball": 0, "pat": 1, "dog": 1, "trol": 1, "anim": 1, "book": 1
}

def main():
    sentences_df = pd.read_csv('G20407.csv')
    template_xml = XMLProcessor('G20406.xml') # do not change this

    events_file = "detected_events.json"
    with open(events_file, 'r') as f:
        events_data = json.load(f)
        events_dict = {event['event']: event for event in events_data}

    print("\nReading XML file...")
    aoi_list, item_dict = template_xml.read_xml()
    print(len(aoi_list))
    
    print("\nReordering entries based on sentences...")
    ordered_aoi_list = template_xml.reorder_by_sentences(item_dict, sentences_df, events_dict, flipped_origin_dict_en)

    template_xml.write_xml(ordered_aoi_list, 'output31.xml')

if __name__ == "__main__":
    main()