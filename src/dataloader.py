from __future__ import annotations
from torch.utils.data import IterableDataset
import os
from pathlib import Path
import pandas as pd
import shutil
import random

file_path_abs = os.path.abspath(__file__)
ws_path = Path(file_path_abs).parents[1]
prews_path = Path(file_path_abs).parents[2]
attr_path = os.path.join(ws_path, 'data', 'M_attr', 'M_attr') 
data_dir = os.path.join(ws_path, 'data', 'UAV-benchmark-M')
gt_dir = os.path.join(ws_path, 'data', 'UAV-benchmark-MOTD_v1.0', 'GT') # ground truth
gt_columns = ["frame_index","target_id","bbox_left","bbox_top","bbox_width","bbox_height","score","in-view","occlusion"]

img_lenx = 1024
img_leny = 540

def generate_yolov5_dataset(
    data_dir: str = data_dir,
    ground_truth_dir: str = gt_dir,
    target_dir: str=os.path.join(ws_path, 'dataset'),
    train_val_test_split: list[float] = [0.8, 0.1, 0.1],
):
    """Generate custom dataset according to yolo v5 format
       https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#11-create-datasetyaml
    """
    assert sum(train_val_test_split) == 1.0, "train_val_test_split should sum to 1.0"
    target_img_dir = os.path.join(target_dir, 'images')
    target_label_dir = os.path.join(target_dir, 'labels')
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    target_img_id = 1
    for s in os.listdir(ground_truth_dir):
        scene_id = s.split('_')[0] # e.g. M0101
        if s.endswith('gt_whole.txt'):
            gt_pd = pd.read_csv(os.path.join(ground_truth_dir, s), sep=',', header=None)
            gt_pd.columns = gt_columns
            # format see yolov5 tutorial
            gt_pd["class"] = 0
            gt_pd["x_center"] = (gt_pd["bbox_left"] + gt_pd["bbox_width"] / 2) / img_lenx
            gt_pd["y_center"] = (gt_pd["bbox_top"] + gt_pd["bbox_height"] / 2) / img_leny
            gt_pd["width"] = gt_pd["bbox_width"] / img_lenx
            gt_pd["height"] = gt_pd["bbox_height"] / img_leny
            
            for fid in set(gt_pd['frame_index']):
                # one frame can have multiple objects
                subgt_pd = gt_pd[gt_pd['frame_index'] == fid]
                label_pd = subgt_pd[["class", "x_center", "y_center", "width", "height"]]
                
                # read from original dataset
                img_name = "img" + str(fid).zfill(6) + ".jpg"
                img_path = os.path.join(data_dir, scene_id, img_name)
                
                # write to target dataset folder
                target_img_path = os.path.join(target_img_dir, 'img' + str(target_img_id).zfill(6) + ".jpg")
                target_label_path = os.path.join(target_label_dir, 'img' + str(target_img_id).zfill(6) + ".txt")
                if not os.path.exists(target_img_path):
                    shutil.copy(img_path, target_img_path)
                
                if not os.path.exists(target_label_path):
                    label_pd.to_csv(target_label_path, sep=' ', header=False, index=False)
                
                target_img_id += 1

    all_img = ['../dataset/images/'+x for x in os.listdir(target_img_dir)]
    # all_img = os.listdir(target_img_dir)
    random.shuffle(all_img)
    train_img = all_img[:int(len(all_img) * train_val_test_split[0])]
    val_img = all_img[int(len(all_img) * train_val_test_split[0]):int(len(all_img) * (train_val_test_split[0] + train_val_test_split[1]))]
    test_img = all_img[int(len(all_img) * (train_val_test_split[0] + train_val_test_split[1])):]
    with open(os.path.join(target_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_img))
    with open(os.path.join(target_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_img))
    with open(os.path.join(target_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_img))