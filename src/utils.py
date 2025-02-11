import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
from pathlib import Path
import sys
ws_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(os.path.join(ws_path, 'yolov5'))
from yolov5.utils.metrics import box_iou

# 匈牙利匹配
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# 用于计算det中的目标框和卡尔曼滤波追踪的目标框的iou
def iou_batch(det_box, track_box2):
    # 将list转为numpy格式
    det_box = np.array(det_box)
    track_box2 = np.array(track_box2)
    # 再将numpy转为tensor，速度更快
    det_box = torch.from_numpy(det_box)
    # 注意这里box2
    track_box2 = torch.from_numpy(track_box2[:, 0:4])
    # 调库算iou
    iou = box_iou(det_box, track_box2)
    return np.array(iou)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):

    # 初始化时，trackers为空，直接返回空匹配结果
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 这里trackers多一列ID列
    iou_matrix = iou_batch(detections, trackers)
    # [[0.73691421 0.         0.         0.        ]
    #  [0.         0.89356082 0.         0.        ]
    #  [0.         0.         0.76781823 0.        ]]

    if min(iou_matrix.shape) > 0:

        a = (iou_matrix > iou_threshold).astype(np.int32)
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]]

        # print(a.sum(1)): [1 1 1]
        # print(a.sum(0)): [1 1 1 0]
        # a.sum(1)表示将矩阵所有列累加，a.sum(0)表示将矩阵所有行累加

        # 如果大于0.3的位置恰好一一对应，可直接得到匹配结果，否则利用匈牙利算法进行匹配
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:

            matched_indices = np.stack(np.where(a), axis=1)
            # [[0 0]
            #  [1 1]
            #  [2 2]]
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # print(unmatched_detections) : []
    # print(unmatched_trackers) : [3]

    # 匈牙利算法匹配出的结果，未必符合IOU大于0.3，需要再进行一次筛选
    # 如果匹配后的IOU数值依旧很小，则同样默认为未匹配成功
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # print(matches): [[0 0] [1 1] [2 2]]
    # print(np.array(unmatched_detections)): []
    # print(np.array(unmatched_trackers)): [3]

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# 将 [x1,y1,x2,y2] 形式转化为 [center_x,center_y,s,r] 形式
def convert_bbox_to_z(bbox):

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))


# 输入的x是一个7维的状态向量，我们只用前4维的边框信息
# 将 [center_x,center_y,s,r] 形式转化为 [x1,y1,x2,y2] 形式
# s = w * h , r = w / h
def convert_x_to_bbox(x):

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))