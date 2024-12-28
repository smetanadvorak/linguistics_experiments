import sys
import os
from collections import defaultdict

class Dataset:
    def __init__(self, dataset_path):
        print("Loading dataset ...")
        self.components = {'video': True, 'frame': False, 'event': False, 'psychopy': True, 'aoi': False}
        self.extensions = [ ".mkv", ".json", ".jpg", ".xlsx"]
        self.root_dir = dataset_path
        self.dirs = {}
        for component in self.components:
            path = os.path.join(dataset_path, component)
            if not os.path.exists(path):
                print(f"Could not find {component} folder at {path}")
                exit()

            self.dirs[component] = path

        sample_fun = lambda : {c: None for c, v in self.components.items()}
        self.data = defaultdict(sample_fun)

        # get input file paths
        for component in self.components:
            if not self.components[component]:
                continue
            for path in self.get_path_list(component):
                id = self.get_file_id(path)
                self.data[id][component] = path

        # check if all input files are present for all subjects
        for id, sample in self.data.items():
            for component, v in sample.items():
                if v is None and self.components[component]:
                    print(f"Subject {id}: missing {component} file")

        # generate intermediate/output file paths
        for id, sample in self.data.items():
            sample['event'] = os.path.join(self.root_dir, 'event', id + ".json")
            sample['aoi'] = os.path.join(self.root_dir, 'aoi', id + ".xml")

        print("Done")


    def get_file_id(self, path):
        id = os.path.basename(path)
        id = id.replace("-scrrec.mkv", "")
        id = id.replace("_TxC.xlsx", "")
        id = id.replace("EN", "")
        id = id.replace("FR", "")
        id = id.replace("RU", "")
        return id

    def get_path_list(self, component):
        if component not in self.components:
            print("Unknown component", component)
            exit()
        path_list = [f for f in os.listdir(self.dirs[component]) if os.path.splitext(f)[1] in self.extensions and not os.path.basename(f).startswith('_')]
        return [os.path.join(self.dirs[component], f) for f in path_list]

    def get_path(self, id, component):
        return self.data[id][component]

    def all_inputs_available(self, id):
        sample = self.data[id]
        if sample['video'] is None or sample['psychopy'] is None:
            return False
        return True

    def events_available(self, id):
        path = self.data[id]['event']
        return path is not None and os.path.exists(path)

    def psychopy_available(self, id):
        path = self.data[id]['psychopy']
        return path is not None and os.path.exists(path)

    def video_available(self, id):
        path = self.data[id]['video']
        return path is not None and os.path.exists(path)

if __name__ == "__main__":
    path = sys.argv[1]
    dataset = Dataset(path)
    # [print(f"{k}: {v}") for k, v in dataset.__dict__.items()]

    # print(dataset.get_videos())