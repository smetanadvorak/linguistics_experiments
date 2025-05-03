import argparse
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
import copy

from eyeoi.babibo.aoi import XMLProcessor
from eyeoi.babibo.babibo_experiment import load_reference, read_experiment_csv
from eyeoi.dataset import Dataset
from eyeoi.frame_extractor import RegionConfig
from eyeoi.event_matcher import VideoEventMatcher


def main():
    parser = argparse.ArgumentParser(description='Process multiple videos for events')
    parser.add_argument('data_dir', help='Directory the dataset (e.g. en, fr)')
    args = parser.parse_args()

    aoi_ref_dict = load_reference()
    
    dataset = Dataset(args.data_dir)

    for id in tqdm(dataset.data):
        if not dataset.psychopy_available(id):
            print("Skipping subject", id, ": missing psychopy file")
            continue

        if not dataset.events_available(id):
            print("Skipping subject", id, ": missing events file")
            continue

        print(os.path.basename(dataset.get_path(id, "psychopy")), os.path.basename(dataset.get_path(id, "event")))

        with open(dataset.get_path(id, "event"), 'r') as f:
            events_data = json.load(f)
            events_list = []
            for i in range(len(events_data)):
                if i % 4 != 2 and i % 4 != 3:
                    events_list.append(events_data[i])

        experiment_df = read_experiment_csv(dataset.get_path(id, "psychopy"))

        ordered_aoi_list = []
        if (len(experiment_df) != len(events_list)):
            print(f"Experiment and events list length mismatch: {len(experiment_df)} vs {len(events_list)}")
        
        for event_idx, (row, event) in enumerate(zip(experiment_df, events_list)):
            video_l = row["Video_LEFT"]
            video_r = row["Video_RIGHT"]
            video_f = row["Video_name"]
            # print(f"f: {video_f}, l: {video_l}:, r: {video_r}")
            aoi_l = copy.deepcopy(aoi_ref_dict[video_f][video_l]["LEFT"])
            aoi_r = copy.deepcopy(aoi_ref_dict[video_f][video_r]["RIGHT"])

            item_name = event['event']
            item_t0 = event['start_time'] - 0.2
            item_t1 = event['end_time'] - 0.2

            if item_name != video_f:
                msg=f"Item name in event and in experiment didn't match: {item_name} and {video_f}"
                raise Exception(msg)
            
            # aoi_l = XMLProcessor.advance_aoi_time(aoi_l, item_t0) 
            # aoi_r = XMLProcessor.advance_aoi_time(aoi_r, item_t0)
            aoi_l = XMLProcessor.set_aoi_start_stop(aoi_l, item_t0, item_t1)
            aoi_r = XMLProcessor.set_aoi_start_stop(aoi_r, item_t0, item_t1)

            ordered_aoi_list.append(aoi_l)
            ordered_aoi_list.append(aoi_r)

        # try:
        output_path = dataset.get_path(id, "aoi")
        _, suf = os.path.splitext(os.path.basename(output_path))
        name, _ = os.path.splitext(os.path.basename(dataset.get_path(id, "psychopy")))
        output_path = os.path.join(os.path.dirname(output_path), name + "_AOIs" + suf)
        XMLProcessor.write_xml(ordered_aoi_list, output_path)


if __name__ == "__main__":
    main()

'''
ENGLISH:
python3 aoi_batch.py \
/Users/akmbpro/Documents/coding/alina/text_completion/data/en \
/Users/akmbpro/Documents/coding/alina/text_completion/data/en/ground_truth/G10206EN_TxC_AOIs.xml
'''

'''
ENGLISH 250405:
python3 aoi_batch.py data_250405/en data_250405/en/G10207EN-scrrec (AOIs) Total v.2.xml data_250405/en/aoi_total
'''