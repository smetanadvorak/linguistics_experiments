import cv2
import os
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from matplotlib import pyplot as plt

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(str(video_path))
        if not self.video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video is ", total_frames / self.fps, " seconds long")

        self.frame_number = 0
        self.video_ended = False

    def read(self):
        ret, frame = self.video.read()
        if not ret:
            self.video_ended = True
            print("Video ended prematurely")
            return None

        self.frame_number += 1
        self.current_time = self.frame_number / self.fps

        return frame
    
    def release(self):
        self.video.release()


class VideoEventMatcher:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.reference_frames: Dict[str, dict] = {}
        self._current_item_index = 0
        self.item_order: Optional[List[str]] = None
        self.region_config = [0, 1, 0, 1]

    def load_reference_frames(self, reference_dir: Union[str, Path]) -> None:
        """Load reference frames from directory"""
        reference_dir = Path(reference_dir)
        for event_file in reference_dir.iterdir():
            if event_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                img = cv2.imread(str(event_file))
                name = os.path.splitext(os.path.basename(event_file))[0]
                print("Read reference image", name)
                if img is not None:
                    self.reference_frames[name] = {
                        'text': img
                    }
                else:
                    print(f"Warning: Failed to load reference image: {event_file}")

        if not self.reference_frames:
            raise ValueError(f"No valid reference frames found in {reference_dir}")

    def set_item_order(self, items: List[str]) -> None:
        """Set the ordered list of items to guide the matching process"""
        # Append .jpg to each item name
        self.item_order = [f"{item}.jpg" for item in items]
        self._current_item_index = 0

    def set_region(self, config):
        self.region_config = config


    @staticmethod
    def compute_text_similarity(text1, text2):
        """Compute text similarity using OCR-friendly comparison"""
        if text1 is None or text2 is None:
            return 0.0

        try:
            # Convert to grayscale and threshold to binary
            gray1 = cv2.cvtColor(text1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(text2, cv2.COLOR_BGR2GRAY)

            # Ensure images are the same size
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

            _, binary1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)
            _, binary2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)

            # Compare using normalized cross-correlation
            result = cv2.matchTemplate(binary1, binary2, cv2.TM_CCORR_NORMED)
            return float(result.max())

        except Exception as e:
            print(f"Warning: Error computing similarity: {str(e)}")
            return 0.0

    @staticmethod
    def extract_regions(frame, config):
        """Extract text region from frame"""
        if frame is None:
            return {'text': None}

        height, width = frame.shape[:2]

        top = int(height * config.top)
        bottom = int(height * config.bottom)
        left = int(width * config.left)
        right = int(width * config.right)

        # Extract region
        region = frame[top:bottom, left:right].copy()

        # Clear timer area if specified
        if config.clear_timer:
            timer_h = config.timer_height if config.timer_height else height // 10
            timer_w = config.timer_width if config.timer_width else width // 10
            region[0:timer_h, 0:timer_w] = 255

        regions = {'text': region}
        return regions

    def find_events_hinted(self, video_path, hint_list):
        """Find events in video using text similarity"""

        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video is ", total_frames / fps, " seconds long")

        detected_events = []

        frame_number = 0
        video_ended = False
        for hint_idx, hint in enumerate(hint_list):
            if video_ended:
                break

            item = hint['item']
            frame_name = item + ".jpg"

            if frame_name not in self.reference_frames:
                print("skipping control:", item)
                continue

            print("processing item", item)

            # start = hint['start'] - 5.0
            # stop = hint['stop'] + 5.0
            # if hint_idx < len(hint_list) - 1:  # if not last item
            #     stop = min(stop, hint_list[hint_idx+1]['start'])

            current_time = frame_number / fps

            # print("fast-forwarding to t", start)
            # while current_time < start:
            #     video.read()
            #     frame_number += 1
            #     current_time = frame_number / fps

            # print("will look until t", stop)

            item_started = False
            item_done = False
            current_event = {}

            while not item_done:
                ret, frame = video.read()
                if not ret:
                    video_ended = True
                    print("Video ended prematurely")

                if video_ended:
                    break

                frame_number += 1
                current_time = frame_number / fps
                regions = self.extract_regions(frame, self.region_config)

                text_sim = self.compute_text_similarity(
                    regions['text'],
                    self.reference_frames[frame_name]['text']
                )

                # print(current_time, item, text_sim)

                # if (current_time > 270 and current_time < 272):
                #     plt.subplot(121)
                #     plt.imshow(regions['text'])
                #     plt.subplot(122)
                #     plt.imshow(self.reference_frames[frame_name]['text'])
                #     plt.savefig('test.jpg')
                #     input('hery')

                if text_sim > self.threshold:
                    if not item_started:
                        print("Start:", item)
                        current_event['start'] = current_time
                        current_event['score'] = text_sim
                        item_started = True
                    else:
                        current_event['score'] += text_sim

                if text_sim < self.threshold and item_started:
                    print("End:", item)
                    event = {
                        'event': item,
                        'start_time': float(current_event['start']),
                        'end_time': float(current_time),
                        'confidence': float(current_event['score'] / ((current_time - current_event['start']) * fps))
                    }
                    detected_events.append(event)
                    item_done = True

        video.release()
        detected_events.sort(key=lambda x: x['start_time'])
        return detected_events




    def find_events(self, video_path: Union[str, Path]) -> List[dict]:
        """Find events in video using text similarity"""
        if not self.reference_frames:
            raise ValueError("No reference frames provided")

        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video is ", total_frames / fps, " seconds long")

        current_events = defaultdict(lambda: {'start': None, 'score': 0})
        detected_events = []

        with tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}") as pbar:
            frame_number = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                current_time = frame_number / fps
                frame_number += 1
                pbar.update(1)

                regions = self.extract_regions(frame, self.region_config)
                if regions['text'] is None:
                    continue

                # Get current event to check
                current_event = self._get_current_event()
                if current_event is None:
                    break  # All events found, stop processing

                if current_event not in self.reference_frames:
                    self._current_item_index += 1  # Skip missing reference frames
                    continue

                text_sim = self.compute_text_similarity(
                    regions['text'],
                    self.reference_frames[current_event]['text']
                )

                if text_sim > self.threshold:
                    if current_events[current_event]['start'] is None:
                        print("Start:", current_event)
                        current_events[current_event]['start'] = current_time
                    current_events[current_event]['score'] += text_sim
                else:
                    if current_events[current_event]['start'] is not None:
                        # Strip .jpg from event name in output
                        print("End:", current_event)
                        event_name = Path(current_event).stem
                        event = {
                            'event': event_name,
                            'start_time': float(current_events[current_event]['start']),
                            'end_time': float(current_time),
                            'confidence': float(current_events[current_event]['score'] /
                                        ((current_time - current_events[current_event]['start']) * fps))
                        }
                        detected_events.append(event)
                        current_events[current_event] = {'start': None, 'score': 0}
                        self._current_item_index += 1  # Move to next event

        video.release()
        detected_events.sort(key=lambda x: x['start_time'])
        return detected_events

    def extract_event(self, video, item_name):
        item_done = False
        current_event = {}
        while not item_done:
            frame = video.read()

            if video.video_ended:
                break

            regions = self.extract_regions(frame, self.region_config)

            sim = self.compute_text_similarity(
                regions['text'],
                self.reference_frames[item_name]['text']
            )
            if sim > self.threshold:
                if not item_started:
                    print("Start:", item_name)
                    current_event['start'] = video.current_time
                    current_event['score'] = sim
                    item_started = True
                else:
                    current_event['score'] += sim

            if sim < self.threshold and item_started:
                print("End:", item_name)
                event = {
                    'event': item_name,
                    'start_time': float(current_event['start']),
                    'end_time': float(video.current_time),
                    'confidence': float(current_event['score'] / ((video.current_time - current_event['start']) * video.fps))
                }
                item_done = True
            
        return current_event


    def find_cross(self, video):
        '''
        Find next cross and skip until right after it.
        '''
        cross_started = False
        cross_ended = False

        while not video.video_ended and not cross_ended:
            frame = video.read()
            regions = self.extract_regions(frame, self.region_config)
            sim = self.compute_text_similarity(
                regions['text'],
                self.reference_frames['cross']['text']
            )
            if sim > self.threshold and not cross_started:
                print("Start cross")
                cross_started = True

            if sim < self.threshold and cross_started:
                print("End cross")
                cross_started = False
                cross_ended = True
                return

    def find_events_cross(self, video_path, item_list):
        """Find occurrences of the pre-video cross on the screen"""

        video = Video(str(video_path))

        detected_events = []
        n_cross_skip = 2 # skip first two crosses as they are the tutorial ones

        event_list = []
        current_event = {}

        for i in range(n_cross_skip):
            self.find_cross(video)

        for item_idx in range(len(item_list)):
            self.find_cross(video)
            item_name = item_list[item_idx]
            item_started = False
            item_ended = False

            print("Item", item_idx, item_name)
            while not video.video_ended and not item_ended:
                frame = video.read()
                regions = self.extract_regions(frame, self.region_config)
                
                level = np.mean(regions['text'].flatten())
                if level < 254.9:
                    if not item_started:
                        print("Start:", item_name)
                        current_event['start'] = video.current_time
                        item_started = True
                        item_ended = False

                if level >= 254.9 and item_started:
                    print("End:", item_name)
                    event = {
                        'event': item_name,
                        'start_time': float(current_event['start']),
                        'end_time': float(video.current_time)
                    }
                    event_list.append(event)
                    item_started = False
                    item_ended = True

        video.release()
        event_list.sort(key=lambda x: x['start_time'])
        return event_list


    def _get_current_event(self) -> Optional[str]:
        """Get the current event to look for"""
        if self.item_order is None:
            return list(self.reference_frames.keys())[0]

        if self._current_item_index >= len(self.item_order):
            return None

        return self.item_order[self._current_item_index]

    def reset_search(self) -> None:
        """Reset the search position to start from the beginning of ordered items"""
        self._current_item_index = 0

def process_video(video_path: Union[str, Path],
                 matcher: VideoEventMatcher,
                 output_path: Union[str, Path]) -> Optional[Path]:
    """Process a single video file"""
    try:
        video_name = Path(video_path).stem
        output_file = Path(output_path) / f"{video_name}_events.json"
        if video_name.endswith('_events'):
            output_file = Path(output_path) / f"{video_name}.json"

        detected_events = matcher.find_events(video_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(detected_events, f, indent=2)

        return output_file
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find events in videos')
    parser.add_argument('input_path', help='Video file or folder of videos')
    parser.add_argument('reference_dir', help='Directory with reference frames')
    parser.add_argument('output_path', help='Path for detected events JSON')
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Similarity threshold (default: 0.8)')
    parser.add_argument('--item-order', nargs='+',
                      help='Optional ordered list of items to guide matching')

    args = parser.parse_args()

    try:
        # Initialize matcher
        matcher = VideoEventMatcher(threshold=args.threshold)
        matcher.load_reference_frames(args.reference_dir)

        # Set item order if provided
        if args.item_order:
            matcher.set_item_order(args.item_order)

        input_path = Path(args.input_path)

        if input_path.is_file():
            output_file = process_video(input_path, matcher, args.output_path)
            if output_file:
                print(f"Saved events to: {output_file}")

        elif input_path.is_dir():
            video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi')) + \
                         list(input_path.glob('*.mkv')) + list(input_path.glob('*.mov'))

            if not video_files:
                print(f"No video files found in {input_path}")
                return

            for video_file in tqdm(video_files, desc="Processing videos"):
                output_file = process_video(video_file, matcher, args.output_path)

        else:
            print(f"Error: '{input_path}' is not a valid file or directory")
            exit(1)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()