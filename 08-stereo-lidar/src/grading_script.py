import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("gt_file", type=Path, default=Path("ground_truth.csv"))
parser.add_argument("pred_dir", type=Path, default=Path("perception_results"))
args = parser.parse_args()

try:
    gt = pd.read_csv(args.gt_file)
except FileNotFoundError:
    print("Ground truth file not found at: ", args.gt_file)
    exit(1)

if not args.pred_dir.is_dir():
    print("Predictions directory not found at: ", args.pred_dir)
    exit(1)
elif len(list(args.pred_dir.glob("*.csv"))) == 0:
    print("No CSV files found in", args.pred_dir)
    exit(1)

# %%
start_index = gt["Frame"].min()
end_index = gt["Frame"].max()


# %% Helpers
def Find_3D_Distance(pos_a, pos_b):
    # Calculate the 3D distance between two points
    return np.sqrt(
        (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2 + (pos_a[2] - pos_b[2]) ** 2
    )


def Find_MVec_Diff(mvec_a, mvec_b):
    return (mvec_a[0] - mvec_b[0]) + (mvec_a[1] - mvec_b[1]) + (mvec_a[2] - mvec_b[2])


def Find_Nearest_Neighbor(pose, set_of_poses):
    # Find the nearest neighbor to a given pose
    min_dist = 1000000000000
    min_index = -1
    for i in range(len(set_of_poses)):
        dist = Find_3D_Distance(pose, set_of_poses[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_dist, min_index


def Get_Position_Score(dist):
    if dist < 1.0:
        return 4
    elif dist < 1.5:
        return 3.8
    elif dist < 2.0:
        return 3.0
    else:
        return 0.0


def Get_MVec_Score(mvec_dist):
    if mvec_dist < 0.5:
        return 3
    elif mvec_dist < 1.0:
        return 1.5
    else:
        return 0.0


# %% [markdown]
# # Plan
# 80% for accuracy
# - 40% for position
#     - 40% if within 1.0m
#     - 38% if within 1.5m
#     - 30% if within 2.0m
#     - 20% if within 3.0m
#     - 0% otherwise
# - 30% for motion vector
#     - 30% if within 0.5
#     - 15% if within 1.0
#     - 0% otherwise
# - 10% for bounding box
#     - 10%

# %%

# Constants
NUMBER_OF_VEHICLES_PER_FRAME = 6
MAX_FRAME_POSITION_SCORE = 4.0 * NUMBER_OF_VEHICLES_PER_FRAME
MAX_FRAME_MVEC_SCORE = 3.0 * NUMBER_OF_VEHICLES_PER_FRAME
MAX_FRAME_BBOX_SCORE = 1.0 * NUMBER_OF_VEHICLES_PER_FRAME
MAX_FRAME_SCORE = MAX_FRAME_POSITION_SCORE + MAX_FRAME_MVEC_SCORE + MAX_FRAME_BBOX_SCORE
NUMBER_OF_FRAMES = end_index - start_index + 1

POSITION_SCORE_PERCENTAGE = 40
MVEC_SCORE_PERCENTAGE = 30
BBOX_SCORE_PERCENTAGE = 10

MAX_SCORE = MAX_FRAME_SCORE * NUMBER_OF_FRAMES
MAX_POSITION_SCORE = MAX_FRAME_POSITION_SCORE * NUMBER_OF_FRAMES
MAX_MVEC_SCORE = MAX_FRAME_MVEC_SCORE * NUMBER_OF_FRAMES
MAX_BBOX_SCORE = MAX_FRAME_BBOX_SCORE * NUMBER_OF_FRAMES


# Lists
position_frame_score = []
mvec_frame_score = []
bbox_frame_score = []
frame_score = []


for frame_number in range(start_index, end_index):
    csv_file = args.pred_dir / f"{frame_number}.csv"
    try:
        pred_frame = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"No file for frame [{frame_number}] at: ", csv_file)
        position_frame_score.append(0)
        mvec_frame_score.append(0)
        bbox_frame_score.append(0)
        frame_score.append(0)
        continue

    # Get all rows in GT and pred for this frame
    gt_frame = gt[gt["Frame"] == frame_number]

    pos_score = 0
    mvec_score = 0
    bbox_score = 0

    false_detections = 0

    if pred_frame.shape[0] != gt_frame.shape[0]:
        # print ("Not equal")
        if pred_frame.empty:
            total_score = 0
            frame_score.append(total_score)
            position_frame_score.append(0)
            mvec_frame_score.append(0)
            bbox_frame_score.append(0)
            print(
                "frame: {}, score: {}/{}, pos: {}/{}, mvec: {}/{}, bbox: {}/{}".format(
                    frame_number,
                    total_score,
                    MAX_FRAME_SCORE,
                    0,
                    MAX_FRAME_POSITION_SCORE,
                    0,
                    MAX_FRAME_MVEC_SCORE,
                    0,
                    MAX_FRAME_MVEC_SCORE,
                )
            )
            continue

        if pred_frame.shape[0] > gt_frame.shape[0]:
            false_detections = pred_frame.shape[0] - gt_frame.shape[0]
            # print ("frame: {}, score: {}/{}, pos: {}/{}, mvec: {}/{}, bbox: {}/{}"\
            #    .format ( frame_number, 0, MAX_FRAME_SCORE, \
            # 0, MAX_FRAME_POSITION_SCORE, 0, MAX_FRAME_MVEC_SCORE, 0, MAX_FRAME_MVEC_SCORE) )
            # continue

    gt_poses = []
    gt_mvecs = []
    for _, row in gt_frame.iterrows():
        gt_pose = [row["Pos_X"], row["Pos_Y"], row["Pos_Z"]]
        gt_mvec = [row["MVec_X"], row["MVec_Y"], row["MVec_Z"]]
        gt_poses.append(gt_pose)
        gt_mvecs.append(gt_mvec)

    pred_poses = []
    pred_mvecs = []
    for _, row in pred_frame.iterrows():
        pred_pose = [row["position_x"], row["position_y"], row["position_z"]]
        pred_mvec = [row["mvec_x"], row["mvec_y"], row["mvec_z"]]
        pred_poses.append(pred_pose)
        pred_mvecs.append(pred_mvec)

    pred_counter = 0
    while pred_counter < len(pred_poses):
        dist, index = Find_Nearest_Neighbor(pred_poses[pred_counter], gt_poses)
        if index == -1:
            pred_counter += 1
            continue
        mvec_dist = Find_MVec_Diff(pred_mvecs[pred_counter], gt_mvecs[index])
        nn = gt_poses[index]

        # Scoring
        pos_score += Get_Position_Score(dist)
        mvec_score += Get_MVec_Score(mvec_dist)
        bbox_score += 1.0

        gt_poses.pop(index)
        pred_counter += 1

    if false_detections > 0:
        print("False detections: ", false_detections)
        pos_score -= false_detections * 2
        mvec_score -= false_detections * 1.5
        bbox_score -= false_detections * 0.5

    position_frame_score.append(pos_score)
    mvec_frame_score.append(mvec_score)
    bbox_frame_score.append(bbox_score)
    total_score = pos_score + mvec_score + bbox_score
    frame_score.append(total_score)
    print(
        "frame [{}], score: {:.2f}/{}, pos: {:.2f}/{}, mvec: {:.2f}/{}, bbox: {:.2f}/{}".format(
            frame_number,
            total_score,
            MAX_FRAME_SCORE,
            pos_score,
            MAX_FRAME_POSITION_SCORE,
            mvec_score,
            MAX_FRAME_MVEC_SCORE,
            bbox_score,
            MAX_FRAME_BBOX_SCORE,
        )
    )

print("***************")
print("***************")

# Print the final score
print(
    "Position score: {:.2f} / {}".format(
        (sum(position_frame_score) / MAX_POSITION_SCORE) * POSITION_SCORE_PERCENTAGE,
        POSITION_SCORE_PERCENTAGE,
    )
)
print(
    "MVec score: {:.2f} / {}".format(
        (sum(mvec_frame_score) / MAX_MVEC_SCORE) * MVEC_SCORE_PERCENTAGE, MVEC_SCORE_PERCENTAGE
    )
)
print(
    "BBox score: {:.2f} / {}".format(
        (sum(bbox_frame_score) / MAX_BBOX_SCORE) * BBOX_SCORE_PERCENTAGE, BBOX_SCORE_PERCENTAGE
    )
)
print(
    "Total score: {:.2f} / {}".format(
        (sum(frame_score) / MAX_SCORE)
        * (POSITION_SCORE_PERCENTAGE + BBOX_SCORE_PERCENTAGE + MVEC_SCORE_PERCENTAGE),
        (POSITION_SCORE_PERCENTAGE + BBOX_SCORE_PERCENTAGE + MVEC_SCORE_PERCENTAGE),
    )
)
