import sys
import argparse
import numpy as np
import os
import datetime

def make_parser():
    parser = argparse.ArgumentParser("Interpolation!")
    parser.add_argument("--fpath", default="", help="tracking results file path")
    parser.add_argument("--n_min", type=int, default=5, help="minimum")
    parser.add_argument("--n_dti", type=int, default=20, help="dti")
    parser.add_argument("--distance_thres", type=int, default=500, help="distance threshold for interpolation")
    return parser

def bbox_distance(right_box, left_box):
    x_r,y_r = right_box[0]+right_box[2]/2,right_box[1]+right_box[3]/2
    x_l,y_l = left_box[0]+left_box[2]/2,left_box[1]+left_box[3]/2
    return ((x_r-x_l)**2+(y_r-y_l)**2)**(0.5)

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def write_results(output_fpath, results):
    save_format = '{frame} {id} {x1} {y1} {w} {h} {s} -1 -1 -1\n'
    lines = []
    for i in range(results.shape[0]):
        frame_data = results[i]
        frame_id = int(frame_data[0])
        track_id = int(frame_data[1])
        x1, y1, w, h = frame_data[2:6]
        score = frame_data[6]
        line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=score)
        lines.append(line)
    sorted_lines = sorted(lines, key=lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1])))
    with open(output_fpath, 'w') as f:
        f.writelines(sorted_lines)

def dti(txt_path, n_min=25, n_dti=20, distance_thres=500):
    large_distance_cnt = 0
    seq_data = np.loadtxt(txt_path, dtype=np.float64, delimiter=' ')
    min_id = int(np.min(seq_data[:, 1]))
    max_id = int(np.max(seq_data[:, 1]))
    print(f"Instances before interpolation: {len(seq_data)}")
    
    seq_data = seq_data[np.lexsort((seq_data[:, 1], seq_data[:, 0]))]
    _, unique_indices, inverse_indices = np.unique(seq_data[:, [0, 1]], axis=0, return_index=True, return_inverse=True)
    counts = np.bincount(inverse_indices)
    seq_data = seq_data[np.where(counts[inverse_indices] == 1)]
    print(len(seq_data))
    
    seq_results = np.zeros((1, 10), dtype=np.float64)
    for track_id in range(min_id, max_id + 1):
        index = (seq_data[:, 1] == track_id)
        tracklet = seq_data[index]
        tracklet_dti = tracklet
        if tracklet.shape[0] == 0:
            continue
        n_frame = tracklet.shape[0]
        # n_conf = np.sum(tracklet[:, 6] > 0.5)
        
        if n_frame > n_min:
            frames = tracklet[:, 0]
            frames_dti = {}
            for i in range(0, n_frame):
                right_frame = frames[i]
                if i > 0:
                    left_frame = frames[i - 1]
                else:
                    left_frame = frames[i]
                # disconnected track interpolation
                if 1 < right_frame - left_frame < n_dti:
                    num_bi = int(right_frame - left_frame - 1)
                    right_bbox = tracklet[i, 2:6]
                    left_bbox = tracklet[i - 1, 2:6]
                    
                    if bbox_distance(right_bbox,left_bbox)>distance_thres:
                        large_distance_cnt += 1
                        continue
                    
                    for j in range(1, num_bi + 1):
                        curr_frame = j + left_frame
                        curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                    (right_frame - left_frame) + left_bbox
                        frames_dti[curr_frame] = curr_bbox
            num_dti = len(frames_dti.keys())
            if num_dti > 0:
                data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                for n in range(num_dti):
                    data_dti[n, 0] = list(frames_dti.keys())[n]
                    data_dti[n, 1] = track_id
                    data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                    data_dti[n, 6:] = [1, -1, -1, -1]
                tracklet_dti = np.vstack((tracklet, data_dti))
        seq_results = np.vstack((seq_results, tracklet_dti))

    seq_results = seq_results[1:]
    seq_results = seq_results[seq_results[:, 0].argsort()]
    print(f"Instances after interpolation: {len(seq_results)}")
    print('large distance count:{}'.format(large_distance_cnt))
    return seq_results

if __name__ == '__main__':
    args = make_parser().parse_args()
    parent_dir = os.path.dirname(args.fpath)
    input_fname = os.path.basename(args.fpath)
    results = dti(args.fpath, n_min=args.n_min, n_dti=args.n_dti, distance_thres=args.distance_thres)
    output_fpath = os.path.join(parent_dir, f'interpolated_{input_fname}')
    write_results(output_fpath, results)