"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from .utils import associate_detections_to_trackers, convert_bbox_to_z, convert_x_to_bbox

np.random.seed(0)


class KalmanBoxTracker(object):
    count = 1

    # 利用bounding box初始化Kalman滤波轨迹
    def __init__(self, bbox, cls):

        self.cls = cls
        # 定义恒定速度模型，7个状态变量和4个观测输入
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状态向量 X = [检测框中心的横坐标，检测框中心的纵坐标，检测框的面积，长宽比，横坐标速度，纵坐标速度，面积速度]
        # SORT假设一个物体在不同帧中检测框的长宽比不变，是个常数，所以速度变化只考虑横坐标、横坐标、检测框面积
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # R是测量噪声的协方差矩阵，即真实值与测量值差的协方差
        # R = diagonal([1, 1, 10, 10])
        self.kf.R[2:, 2:] *= 10.
        # [[ 1.  0.  0.  0.]
        #  [ 0.  1.  0.  0.]
        #  [ 0.  0. 10.  0.]
        #  [ 0.  0.  0. 10.]]

        # P是先验估计的协方差，对不可观测的初始速度，给予高度不确定性
        # P = diagonal([10，10，10，10，1000，1000，1000])
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        # [[   10.     0.     0.     0.     0.     0.     0.]
        #  [    0.    10.     0.     0.     0.     0.     0.]
        #  [    0.     0.    10.     0.     0.     0.     0.]
        #  [    0.     0.     0.    10.     0.     0.     0.]
        #  [    0.     0.     0.     0. 10000.     0.     0.]
        #  [    0.     0.     0.     0.     0. 10000.     0.]
        #  [    0.     0.     0.     0.     0.     0. 10000.]]

        # Q是系统状态变换误差的协方差
        # Q = diagonal([1, 1, 1, 1, 0.01, 0.01, 0.0001])
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # [[1.e+00 0.e+00 0.e+00 0.e+00 0.e+00 0.e+00 0.e+00]
        #  [0.e+00 1.e+00 0.e+00 0.e+00 0.e+00 0.e+00 0.e+00]
        #  [0.e+00 0.e+00 1.e+00 0.e+00 0.e+00 0.e+00 0.e+00]
        #  [0.e+00 0.e+00 0.e+00 1.e+00 0.e+00 0.e+00 0.e+00]
        #  [0.e+00 0.e+00 0.e+00 0.e+00 1.e-02 0.e+00 0.e+00]
        #  [0.e+00 0.e+00 0.e+00 0.e+00 0.e+00 1.e-02 0.e+00]
        #  [0.e+00 0.e+00 0.e+00 0.e+00 0.e+00 0.e+00 1.e-04]]

        # Kalman滤波器初始化时，直接用第一次观测结果赋值状态信息
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # 自上次未匹配成功，经过的帧数
        self.time_since_update = 0

        # 每次创建新的kalman滤波器时，计数ID都会加1
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 存储历史时刻的Kalman状态
        self.history = []

        # 自上次未匹配成功，连续成功匹配的帧数
        self.hit_streak = 0

    def update(self, bbox):

        # 利用已经观测到的边框信息，更新状态向量，这里直接调用了kalman库函数，没有自己手动实现Kalman迭代公式

        # 重置，每次匹配成功，则会调用update函数，即自上次未匹配成功，经过的帧数变为了0
        self.time_since_update = 0

        # 清空history
        # 如果匹配成功，则会调用update函数，history会被重置清空，后面会调用predict函数，将最新的状态添加入history列表
        self.history = []

        # 表示连续匹配成功的次数加一
        self.hit_streak += 1

        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):

        # 如果边界框面积+面积变化速度<=0，就将面积变化速度赋值为0
        # 因为下一时刻边框面积数值，就等于边界框面积+面积变化速度，这样处理可以防止出现面积小于0的情况
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()

        # 一旦出现未成功匹配，则连续匹配上的帧数被置0
        if self.time_since_update > 0:
            self.hit_streak = 0

        # 每经过一帧，都会执行predict函数，time_since_update都会加1
        # 但每一帧只有成功匹配上时，才会执行update函数，time_since_update又会被置0
        # 所以time_since_update表示，自上次未成功匹配，所经过的帧数
        self.time_since_update += 1

        # 将当前Kalman状态存入history历史信息列表
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):

        # 返回当前边界框估计值
        return convert_x_to_bbox(self.kf.x)



class Sort(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):

        self.max_age = max_age    # 1
        self.min_hits = min_hits    # 3
        self.iou_threshold = iou_threshold    # 0.3

        # trackers列表中存储的每一个元素，都是一个特定目标物体的Kalman滤波过程
        self.trackers = []

    def update(self, dets=np.empty((0, 4)), cls=np.empty(0)):

        # det_s是当前帧目标检测结果，默认初始值为np.empty((0, 5))，代表当前帧没有检测到任何目标物体
        # trk_s存储Kalman滤波器对当前帧的位置预测结果

        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        # ret是最终返回值，存储当前帧的目标跟踪结果
        ret = []

        # 类似c++中的引用，每次trk_s中的trk值被修改，trk_s整个数组也会被修改
        for t, trk in enumerate(trks):

            # 对每个轨迹调用Kalman滤波predict函数，预测当前帧的位置
            pos = self.trackers[t].predict()[0]
            # [286.552 154.138 357.889 321.466]

            trk[:] = [pos[0], pos[1], pos[2], pos[3], 1]

            # np.isnan()是判断是否是空值
            # np.any，或的关系，只要有一个满足，则输出为TRUE
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # np.ma.masked_invalid函数：当数组a中某些元素为无效值(NaNs or inf)时，则将无效值的元素设置为mask(–)
        # np.ma.compress_rows函数：返回被屏蔽完成以后的数组
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 将Kalman滤波器预测出来的位置，与当前检测出的位置，进行匹配
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 如果匹配成功，则用观测信息进一步更新Kalman滤波器状态
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # 对于未匹配上的检测结果，默认为出现新轨迹，初始化创建新的Kalman滤波器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i], cls[i])
            self.trackers.append(trk)

        # 对当前tracks列表中的Kalman滤波器依次检验，销毁无效的轨迹，返回当前帧的目标跟踪结果
        i = len(self.trackers)
        for trk in reversed(self.trackers):

            # 以逆序方式，从倒数第一个开始遍历tracks列表中的Kalman滤波器
            d = trk.get_state()[0]

            # trk.time_since_update<1表示，该轨迹状态自上次未被匹配到，经过的帧数小于1，也就是当前帧成功匹配
            # trk.hit_streak>=表示，自上次未被匹配到，连续匹配的帧数大于等于3
            if (trk.time_since_update < 1) and (trk.hit_streak > self.min_hits):
                ret.append(np.concatenate((d, [trk.id], [trk.cls])).reshape(1, -1))

            i -= 1

            # 如果连续未被匹配到大于等于2帧，删除销毁无效的轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:

            # 返回的是，当前帧被认为真正有效的轨迹坐标、对应ID
            # [[542.825 206.921 566.939 251.902   4.   ]
            #  [456.542 213.434 481.675 270.969   3.   ]
            #  [222.571 179.989 256.338 288.916   2.   ]
            #  [286.552 154.138 357.889 321.466   1.   ]]

            return np.concatenate(ret)

        return np.empty((0, 5))
