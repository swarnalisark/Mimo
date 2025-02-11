from __future__ import annotations
import numpy as np

# import os
# from pathlib import Path
# import sys
# ws_path = Path(os.path.abspath(__file__)).parents[1]
# sys.path.append(os.path.join(ws_path, 'yolov5'))

from src.utils import convert_bbox_to_z, convert_x_to_bbox, associate_detections_to_trackers
np.random.seed(0)

class SageHusaFilter:
    def __init__(self,
                 transitionMatrix: np.ndarray,
                 measurementMatrix: np.ndarray,
                 errorCov_init: np.ndarray,
                 processNoiseMean_init: np.ndarray,
                 processNoiseCov_init: np.ndarray,
                 measurementNoiseMean_init: np.ndarray,
                 measurementNoiseCov_init: np.ndarray,
                 state_init: np.ndarray,
                 controlMatrix: np.ndarray | None = None,
                 forgettingFactor: float = 0.99, #! if 1, it's equivalent to Kalman Filter
                 ) -> None: 
        
        self.transitionMatrix = transitionMatrix
        self.controlMatrix = controlMatrix
        self.measurementMatrix = measurementMatrix
        
        self.processNoiseMean = processNoiseMean_init
        self.processNoiseCov = processNoiseCov_init
        self.measurementNoiseMean = measurementNoiseMean_init
        self.measurementNoiseCov = measurementNoiseCov_init
        
        self.post_state = state_init
        self.pred_state = np.ones_like(state_init)
        
        self.post_errorCov = errorCov_init
        self.pred_errorCov = np.zeros_like(self.processNoiseCov)
        
        self.kalmanGain = np.zeros_like(self.post_errorCov)
        
        self.forgettingFactor = forgettingFactor
        self._step = 1
        
    def predict(self, control: np.ndarray | None = None) -> np.ndarray:
        
        if control:
            self.pred_state = self.transitionMatrix @ self.post_state + self.controlMatrix @ control + self.processNoiseMean
        else:
            self.pred_state = self.transitionMatrix @ self.post_state + self.processNoiseMean
            
        self.pred_errorCov = self.transitionMatrix @ self.post_errorCov @ self.transitionMatrix.T + self.processNoiseCov
        self.kalmanGain = self.pred_errorCov @ self.measurementMatrix.T @ np.linalg.inv(self.measurementMatrix @ self.pred_errorCov @ self.measurementMatrix.T + self.measurementNoiseCov)
        
        return self.pred_state
    
    def correct(self, measurement: np.ndarray) -> np.ndarray:
        last_state = self.post_state.copy()
        error = measurement - self.measurementMatrix @ self.pred_state - self.measurementNoiseMean
        self.post_state = self.pred_state + self.kalmanGain @ error
        
        last_errorCov = self.post_errorCov.copy()
        self.post_errorCov = (np.eye(self.post_errorCov.shape[0]) - self.kalmanGain @ self.measurementMatrix) @ self.pred_errorCov
        
        factor = (1-self.forgettingFactor) / (1-self.forgettingFactor**self._step+1e-7)
        # factor = 0
        self._step += 1
        
        self.processNoiseMean = (1-factor) * self.processNoiseMean + factor * (self.post_state - self.transitionMatrix @ last_state)
        self.processNoiseCov = (1-factor) * self.processNoiseCov + factor * (self.kalmanGain@error @ error.T@self.kalmanGain.T + self.post_errorCov - self.transitionMatrix @ last_errorCov @ self.transitionMatrix.T)
        
        self.measurementNoiseMean = (1-factor) * self.measurementNoiseMean + factor * (measurement - self.measurementMatrix @ self.pred_state)
        self.measurementNoiseCov = (1-factor) * self.measurementNoiseCov + factor * (error @ error.T - self.measurementMatrix @ self.pred_errorCov @ self.measurementMatrix.T)
        
        return self.post_state

class SageHusaBoxTracker(object):
    count = 1
    
    def __init__(self, bbox, cls) -> None:
        
        self.cls = cls
        
        # 状态向量 X = [检测框中心的横坐标，检测框中心的纵坐标，检测框的面积，长宽比，横坐标速度，纵坐标速度，面积速度]
        # SORT假设一个物体在不同帧中检测框的长宽比不变，是个常数，所以速度变化只考虑横坐标、横坐标、检测框面积
        F = np.array([[1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1]])
        
        H = np.array([[1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0]])
        
        r = np.zeros((4,1))
        # R是测量噪声的协方差矩阵，即真实值与测量值差的协方差
        R = np.diag([1, 1, 10, 10])
        
        # P是先验估计的协方差，对不可观测的初始速度，给予高度不确定性
        P = np.diag([10, 10, 10, 10, 1000, 1000, 1000])
        
        q = np.zeros((7,1))
        # Q是系统状态变换误差的协方差
        Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001])
        
        x = np.ones((7,1))
        x[:4] = convert_bbox_to_z(bbox)
        
        self.sh = SageHusaFilter(transitionMatrix=F, 
                                 measurementMatrix=H, 
                                 errorCov_init=P,
                                 processNoiseMean_init=q,
                                 processNoiseCov_init=Q,
                                 measurementNoiseMean_init=r,
                                 measurementNoiseCov_init=R, 
                                 state_init=x)
        
        # 自上次未匹配成功，经过的帧数
        self.time_since_update = 0
        
        self.id = SageHusaBoxTracker.count
        SageHusaBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        self.sh.correct(convert_bbox_to_z(bbox))
        
    def predict(self):
        if (self.sh.post_state[6] + self.sh.post_state[2] < 0):
            self.sh.post_state[6] *= 0.0
        self.sh.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(convert_x_to_bbox(self.sh.post_state))
        
        return self.history[-1]
    
    def get_state(self):
        return convert_x_to_bbox(self.sh.pred_state)
    

class SHTracker(object):

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
            trk = SageHusaBoxTracker(dets[i], cls[i])
            self.trackers.append(trk)

        # 对当前tracks列表中的Kalman滤波器依次检验，销毁无效的轨迹，返回当前帧的目标跟踪结果
        i = len(self.trackers)
        for trk in reversed(self.trackers):

            # 以逆序方式，从倒数第一个开始遍历tracks列表中的Kalman滤波器
            d = trk.get_state()[0]
            # if np.isnan(d).any():
            #     raise ValueError

            # trk.time_since_update<1表示，该轨迹状态自上次未被匹配到，经过的帧数小于1，也就是当前帧成功匹配
            # trk.hit_streak>=表示，自上次未被匹配到，连续匹配的帧数大于等于3
            if (trk.time_since_update < 1) and (trk.hit_streak > self.min_hits) and (not np.isnan(d).any()):
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
