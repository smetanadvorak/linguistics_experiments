import os
from tqdm import tqdm
import argparse
import json
from eyeoi.dataset import Dataset
from eyeoi.event_matcher import VideoEventMatcher
from eyeoi.tme.text_matching_experiment import get_item_order_time
from eyeoi.frame_extractor import RegionConfig


# LIMIT_IDs = [
#     '_G20624',
#     'G20422',
#     '_G30717',
#     'G10207',
#     'G10212',
#     'G20421',
#     'G20426',
#     'G20608',
#     'G20408',
#     'G20410',
#     'G20417',
#     'G30710',
#     'G20617',
#     'G20610',
#     'G30708',
#     'G20626',
#     'G20621',
#     'G20611',
#     'G20616',
#     'G30709',
#     'G20620',
#     'G20627',
# ]

def main():
    parser = argparse.ArgumentParser(description='Process multiple videos for events')
    parser.add_argument('data_dir', help='Directory the dataset (e.g. en, fr)')
    parser.add_argument('--threshold', type=float, default=0.95,
                      help='Similarity threshold (default: 0.95)')

    args = parser.parse_args()

    dataset = Dataset(args.data_dir)

    region_config = RegionConfig(
        top=0.0, bottom=1/3, left=0, right=1,
        clear_timer=True
    )

    for id in tqdm(dataset.data):
        print("ID: ", id)

        # if id not in LIMIT_IDs:
        #     print("Skipping", id)
        #     continue

        if not dataset.all_inputs_available(id):
            print("Skipping subject", id, ": missing files")
            continue

        print(os.path.basename(dataset.get_path(id, "psychopy")), os.path.basename(dataset.get_path(id, "video")))

        matcher = VideoEventMatcher(threshold=args.threshold)
        matcher.load_reference_frames(dataset.dirs['frame'])
        matcher.set_region(region_config)

        hint_list = get_item_order_time(dataset.get_path(id, "psychopy"))
        detected_events = matcher.find_events_hinted(dataset.get_path(id, "video"), hint_list)

        with open(dataset.get_path(id, "event"), 'w') as f:
            json.dump(detected_events, f, indent=2)

        print(f"Saved events to: {dataset.get_path(id, "event")}")

if __name__ == "__main__":
    main()