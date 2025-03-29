import argparse
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm

from eyeoi.production_task.aoi import *
from eyeoi.dataset import Dataset

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process an input file and three folders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add required input file argument
    parser.add_argument(
        '--template',
        required=True,
        help='Path to the template aoi xml file'
    )

    # Add three required folder arguments
    parser.add_argument(
        '--psycho',
        required=True,
        help='Path to the folder with psychopi csvs'
    )
    parser.add_argument(
        '--events',
        required=True,
        help='Path to the folder with event jsons'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to the output folder'
    )

    args = parser.parse_args()

    # Verify that input file exists
    if not os.path.isfile(args.template):
        parser.error(f'Input file does not exist: {args.template}')

    # Verify that all folders exist
    for folder in [args.psycho, args.events, args.output]:
        if not os.path.isdir(folder):
            parser.error(f'Folder does not exist: {folder}')

    return args


def main():
    parser = argparse.ArgumentParser(description='Process multiple videos for events')
    parser.add_argument('data_dir', help='Directory the dataset (e.g. en, fr)')
    parser.add_argument('template', help='Path to template AOI .xml file')
    args = parser.parse_args()

    dataset = Dataset(args.data_dir)

    for id in tqdm(dataset.data):
        if not dataset.psychopy_available(id):
            print("Skipping subject", id, ": missing psychopy file")
            continue

        if not dataset.events_available(id):
            print("Skipping subject", id, ": missing events file")
            continue

        print(os.path.basename(dataset.get_path(id, "psychopy")), os.path.basename(dataset.get_path(id, "event")))

        template_xml = XMLProcessor(args.template)
        box_dict = template_xml.read_xml()

        with open(dataset.get_path(id, "event"), 'r') as f:
            events_data = json.load(f)
        # try:
        ordered_aoi_list = template_xml.generate_experiment(events_data)
        output_path = dataset.get_path(id, "aoi")
        pre, suf = os.path.splitext(os.path.basename(output_path))
        pre = pre + "EN_PR"
        output_path = os.path.join(os.path.dirname(output_path), pre + "_AOIs" + suf)
        template_xml.write_xml(ordered_aoi_list, output_path)
        # except Exception as e:
        #     print("Couldn't create AOI file")


if __name__ == "__main__":
    main()

'''
ENGLISH:
python3 aoi_batch.py \
/Users/akmbpro/Documents/coding/alina/text_completion/data/en \
/Users/akmbpro/Documents/coding/alina/text_completion/data/en/ground_truth/G10206EN_TxC_AOIs.xml
'''