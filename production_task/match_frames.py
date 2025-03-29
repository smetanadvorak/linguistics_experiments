import os
from tqdm import tqdm
import argparse
import json
from eyeoi.dataset import Dataset
from eyeoi.event_matcher import VideoEventMatcher
from eyeoi.production_task.production_task_experiment import get_item_order
from eyeoi.frame_extractor import RegionConfig

def main():
    parser = argparse.ArgumentParser(description='Process multiple videos for events')
    parser.add_argument('data_dir', help='Directory the dataset (e.g. en, fr)')
    parser.add_argument('--threshold', type=float, default=0.95,
                      help='Similarity threshold (default: 0.95)')

    args = parser.parse_args()

    dataset = Dataset(args.data_dir)

    region_config = RegionConfig(
        top=0.45, bottom=0.55, 
        left=0.45, right=0.55,
        clear_timer=False
    )

    for id in tqdm(dataset.data):
        print("ID: ", id)

        if not dataset.all_inputs_available(id):
            print("Skipping subject", id, ": missing files")
            continue

        print(os.path.basename(dataset.get_path(id, "psychopy")), os.path.basename(dataset.get_path(id, "video")))

        matcher = VideoEventMatcher(threshold=args.threshold)
        matcher.load_reference_frames(dataset.dirs['frame'])
        matcher.set_region(region_config)

        item_order = get_item_order(dataset.get_path(id, "psychopy"))
        detected_events = matcher.find_events_cross(dataset.get_path(id, "video"), item_order)

        with open(dataset.get_path(id, "event"), 'w') as f:
            json.dump(detected_events, f, indent=2)

        print(f"Saved events to: {dataset.get_path(id, "event")}")

if __name__ == "__main__":
    main()