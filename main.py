import os

import numpy as np
import pandas as pd

import feature_extraction

def main():

    batch_size = 100
    if not os.path.exists("processed_data"):
        os.mkdir("processed_data")
    lefthand_folder = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\lefthand_videos"
    righthand_folder = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\righthand_videos"
    lefthand_train_txt = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\lefthand_train.txt"
    righthand_train_txt = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\righthand_train.txt"
    lefthand_val_txt = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\lefthand_val.txt"
    righthand_val_txt = "C:\\Users\\Gianluca\\Desktop\\FT_LSTM\\PDMotorDB\\righthand_val.txt"
    lefthand_data = []
    righthand_data = []

    # Load labels from lefthand_train.txt
    lefthand_labels_train = {}
    with open(lefthand_train_txt, 'r') as file:
        for line in file:
            video_id, label = line.strip().split()
            lefthand_labels_train[video_id] = label

    # Load labels from righthand_train.txt
    righthand_labels_train = {}
    with open(righthand_train_txt, 'r') as file:
        for line in file:
            video_id, label = line.strip().split()
            righthand_labels_train[video_id] = label

    # Load labels from lefthand_val.txt
    lefthand_labels_val = {}
    with open(lefthand_val_txt, 'r') as file:
        for line in file:
            video_id, label = line.strip().split()
            lefthand_labels_val[video_id] = label

    # Load labels from righthand_val.txt
    righthand_labels_val = {}
    with open(righthand_val_txt, 'r') as file:
        for line in file:
            video_id, label = line.strip().split()
            righthand_labels_val[video_id] = label

    count = 0
    # Process lefthand videos
    processed_lefthand_data = pd.DataFrame()
    for avi_file in os.listdir(lefthand_folder):
        if avi_file.endswith(".avi"):
            video_id = avi_file.split(".")[0]
            if int(video_id) in [8, 36, 58, 73, 74, 83, 91, 95, 98, 104, 144, 164, 186, 179, 191, 93, 198, 217, 233, 253, 282, 343, 111, 157, 183,
                                 180, 187, 236, 268, 283, 295, 307, 319, 324, 332, 346, 351]:
                continue
            label = lefthand_labels_train.get(video_id, lefthand_labels_val.get(video_id, None))
            if label is not None:
                avi_path = os.path.join(lefthand_folder, avi_file)
                features = feature_extraction.extract_features(avi_path, "PDMotorDBProcessed\\Left", 'left', (label, ""))
                if features:
                    processed_lefthand_data=pd.concat([processed_lefthand_data, pd.DataFrame([features])], ignore_index=True)
                    count += 1
                    print(str(count) + " video processed so far")
                else:
                    print("no features in " + video_id)
            else:
                print("No label for video " + video_id)

    processed_lefthand_data.to_csv("PDMotorDB_left.csv")

    # Process righthand videos
    processed_righthand_data = pd.DataFrame()

    for avi_file in os.listdir(righthand_folder):
        if avi_file.endswith(".avi"):
            video_id = avi_file.split(".")[0]
            if int(video_id) in [4,8, 17, 35, 46, 88, 115, 116, 194, 218, 62, 73, 197]:
                continue
            label = righthand_labels_train.get(video_id, righthand_labels_val.get(video_id, None))
            if label is not None:
                avi_path = os.path.join(righthand_folder, avi_file)
                features = feature_extraction.extract_features(avi_path, "PDMotorDBProcessed\\Right", 'right', (label, ""))
                if features:
                    processed_righthand_data = pd.concat([processed_righthand_data, pd.DataFrame([features])],
                                                        ignore_index=True)
                    count += 1
                    print(str(count) + " video processed so far")
                else:
                    print("no features found in " + video_id)
            else:
                print("No label for video " + video_id)

    processed_righthand_data.to_csv("PDMotorDB_right.csv")
    return

if __name__=='__main__':
    main()