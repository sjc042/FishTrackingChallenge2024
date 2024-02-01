import glob
import cv2
import os
import numpy as np
import argparse

limit = 10000
process_num = 0
process_limit = 5

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def make_parser():
    parser = argparse.ArgumentParser("Interpolation!")
    parser.add_argument("--video_path", default="", help="video path")
    parser.add_argument("--fpath", default="", help="tracking results file path")
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    parent_dir = os.path.dirname(args.video_path)
    video_in_path = args.video_path
    video_fname = os.path.basename(video_in_path)
    video_out_path = os.path.join(parent_dir, f'annotated_{video_fname}')
    input_fname = os.path.basename(args.fpath)
    result_path = args.fpath
    if os.path.exists(video_out_path):
        os.remove(video_out_path)
    assert(os.path.exists(video_in_path))
    assert(os.path.exists(result_path))

    label = np.genfromtxt(result_path, delimiter=' ')
    print("got here")
    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        print("Error opening video file.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # need to transpose video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    result = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    frame_num = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # all lines relating to current frame
        frame_label = label[label[:, 0] == frame_num]
        # annotation format:
        #   frame_number track_id center_x center_y w h confidence -1 -1 -1
        # NOTE: annotation coordinates might be transposed
        for _, trk_id, y, x, h, w, confidence, _, _, _ in frame_label:
            assert(x <= width and y <= height)
            cv2.putText(frame, f'{int(trk_id)}', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), thickness=3)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), get_color(trk_id), 2)

        result.write(frame)

        if frame_num % 500 == 0:
            print('Processing frame:', frame_num)

        if frame_num == limit:
            break

        frame_num += 1
        
    print(f"Done!\nTotal frames processed: {frame_num}")
    cap.release()
    result.release()
    cv2.destroyAllWindows()