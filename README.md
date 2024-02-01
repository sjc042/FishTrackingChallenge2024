# Fish Tracking Challenge 2024 Repository

This repository contains the Python and Jupyter Notebook files used by our team for the Fish Tracking Challenge 2024. The project aims to track sweetfish in controlled laboratory video settings using deep learning methods.

## Cloning the Repository

To clone this repository, use the following command in your terminal:

```bash
git clone https://github.com/sjc042/FishTrackingChallenge2024.git
```

## Environment Requirements
To run the scripts and notebooks in this repository, you'll need Python 3.8 or higher. It's recommended to use a virtual environment to manage the dependencies. The following packages are required:

- numpy
- opencv-python
- matplotlib
- pandas
- torch
- torchvision
- ultralytics

## Downloading Data
Data for the challenge can be downloaded from the [Fish Challenge 2024 website](https://ftc-2024.github.io/). Please ensure you have registered and accepted the challenge's terms and conditions. The data includes training, development, and testing videos along with ground truth annotations.

## Running the Code
### Preprocessing Dataset
Prepare the dataset containing video sequences and extract all image frames from training and development videos.
### Training
Follow the instructions in train_sweetfish.ipynb for training the detection and tracking model.
### Postprocessing
1. Iterpolate Tracks:
```bash
python iterpolate_tracks.py --fpath {path/to/detection/results.txt}
```
2. Merge Tracks:
```bash
python merge_tracks.py --fpath {path/to/detection/results.txt}
```
3. Visualize Results in Video:
```bash
python viz_results.py --video_path {path/to/video/file.mp4} --fpath {path/to/detection/results.txt}
```
4. Additional Track Continuity Visualization
- The track_viz.ipynb notebook provides additional visualization for track continuity.
