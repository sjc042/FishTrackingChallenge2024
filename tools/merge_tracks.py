import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime
import argparse

class Track:
    def __init__(self, track_id):
        self.track_id = track_id
        self.frames = []
        self.tl_position = []  # top-left position (x, y)
        self.center_position = []  # center position (x+w/2, y+h/2)
        self.wh = []

    def add_frame_data(self, frame, x, y, w, h):
        self.frames.append(frame)
        self.tl_position.append((x, y))
        self.center_position.append((x + w/2, y + h/2))
        self.wh.append((w, h))

def make_parser():
    parser = argparse.ArgumentParser("Merge Tracks")
    parser.add_argument("--fpath", default='', help='tracking result file path')
    parser.add_argument("--MaxFrameGap", default=200, help="Maximum frame gap allowed between two tracks to merge")
    return parser

def parse_tracking_results(filepath):                   # generate track objects from tracking results
    with open(filepath, 'r') as file:
        content = file.readlines()

    tracks = {}
    for line in content:
        split_line = line.strip().split()
        frame_number = int(split_line[0])
        track_id = int(split_line[1])
        x, y, w, h = map(float, split_line[2:6])

        if track_id not in tracks:
            tracks[track_id] = Track(track_id)
        
        tracks[track_id].add_frame_data(frame_number, x, y, w, h)

    return list(tracks.values())

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calculate_distances(track_objects):                 # get distance matrix from all tracks
    def normalize_distance_matrix(distances):           # normalize finite distance values
        valid_distances = distances[np.isfinite(distances)]
        if len(valid_distances) == 0: return distances
        # Calculating min and max of valid distances
        min_distance = np.min(valid_distances)
        max_distance = np.max(valid_distances)
        if np.absolute(min_distance - max_distance) < 0.00001:
            normalized_distances = np.where(
                np.isfinite(distances),
                distances / max_distance,
                distances
            )
        else:
            # Apply min-max normalization, excluding 'inf' values
            normalized_distances = np.where(
                np.isfinite(distances),
                (distances - min_distance) / (max_distance - min_distance),
                distances
            )
        return normalized_distances
    
    # Initialize all distances to infinity
    n = len(track_objects)
    distances = np.full((n, n), float('inf'))

    for i in range(n):
        for j in range(i + 1, n):
            track_1 = track_objects[i]
            track_2 = track_objects[j]

            # Check temporal overlap
            if set(track_1.frames).intersection(set(track_2.frames)):
                continue

            # Check max frame gap and calculate distance
            if track_2.frames[0] > track_1.frames[-1]:  # track 1 appears first
                # max frame gap check
                if track_2.frames[0] - track_1.frames[-1] > args.MaxFrameGap:
                    continue
                # calculate distance based on the center positions
                distance = euclidean_distance(track_2.center_position[0], track_1.center_position[-1])
            else:                                       # track 2 appears first
                if track_1.frames[0] - track_2.frames[-1] > args.MaxFrameGap:
                    continue
                distance = euclidean_distance(track_1.center_position[0], track_2.center_position[-1])
            
            # Update Symmetric matrix
            distances[i][j] = distance
            distances[j][i] = distance
    
    # Normalize distances
    return normalize_distance_matrix(distances)

def merge_tracks(distances, track_objects):
    iter = 0
    while True:
        if not np.any(np.isfinite(distances)):
            print(f"Iteration: {iter} | Tracks Remaining: {len(track_objects)}")
            break
        if iter % 5 == 0:
            print(f"Iteration: {iter} | Tracks Remaining: {len(track_objects)}")
        iter += 1
        
        # Find the minimum distance and its tracks indices in the temporary array
        min_distance_idx = np.argmin(distances)
        indices_to_merge = np.unravel_index(min_distance_idx, distances.shape)

        # Merge track pair
        track_1, track_2 = indices_to_merge
        track_objects[track_1].frames.extend(track_objects[track_2].frames)
        track_objects[track_1].tl_position.extend(track_objects[track_2].tl_position)
        track_objects[track_1].center_position.extend(track_objects[track_2].center_position)
        track_objects[track_1].wh.extend(track_objects[track_2].wh)
        # Remove the merged track
        del track_objects[track_2]

        # Update the distance matrix
        distances = calculate_distances(track_objects)
        # distances_copy = update_distance_matrix_after_merge(distances_copy, track_1, track_2)
    return track_objects, distances

def filter_tracks(tracks, variance_threshold = 99999):
    filtered_tracks = []
    for track in tracks:
        # Calculate variance for x and y positions
        var_x = np.var([pos[0] for pos in track.tl_position])
        var_y = np.var([pos[1] for pos in track.tl_position])
        # Check if either variance is above the threshold
        if var_x > variance_threshold and var_y > variance_threshold:
            filtered_tracks.append(track)
    return filtered_tracks

def write_tracks_to_mot_format(tracks, output_file_path):
    lines = []
    for track in tracks:
        for i in range(len(track.frames)):
            frame = track.frames[i]
            x, y = track.tl_position[i]
            w, h = track.wh[i]
            # MOT format: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
            line = f"{frame} {track.track_id} {x} {y} {w} {h} -1 -1 -1 -1\n"
            # file.write(line)
            lines.append(line)
    sorted_lines = sorted(lines, key=lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1])))
    with open(output_file_path, 'w') as file:
        file.writelines(sorted_lines)

def make_time_dir(data_dir):
    # Get the current date and time
    current_time = datetime.datetime.now()
    # Format the current time as a string (you can customize the format)
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    time_folder_path = os.path.join(data_dir, time_str)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    return time_folder_path

if __name__ == '__main__':
    args = make_parser().parse_args()
    parent_dir = os.path.dirname(args.fpath)
    # Parse the tracking results file and get the list of Track objects
    track_objects = parse_tracking_results(args.fpath)
    # Get initial distance matrix
    distances = calculate_distances(track_objects)
    # Merge tracks
    merged_tracks, _ = merge_tracks(distances, track_objects)
    # Filter low variance tracks
    filtered_tracks = filter_tracks(merged_tracks)
    # TODO: interpolate tracks

    # Output the results
    input_fname = os.path.basename(args.fpath)
    output_fpath = os.path.join(make_time_dir(parent_dir), f'merged_{input_fname}')
    write_tracks_to_mot_format(filtered_tracks, output_fpath)
