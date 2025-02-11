from src.sage_husa import SHTracker
from src.sort import Sort
import numpy as np
import torch
import argparse
import os
import cv2 as cv
import src
from pathlib import Path

ws_path = Path(src.__file__).parent.parent
yolo_path = os.path.join(ws_path, 'yolov5')
model_path = os.path.join(ws_path, 'runs', 'train', 'exp', 'weights', 'best.pt')
data_dir = os.path.join(ws_path, 'data', 'UAV-benchmark-M')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='sage_husa', help='tracking method, sage_husa or sort')
    parser.add_argument('--scene', default='M0210', help='scence name')
    parser.add_argument('--save', default=True, help='save results')
    parser.add_argument('--save_txt', default=False, help='save txt results')
    parser.add_argument('--save_dir', default='data/track', help='save results dir')
    parser.add_argument('--show', default=True, help='show frame img')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_opt()
    colours = np.random.rand(32, 3) * 255
    yolo = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
    if opt.method == 'sage_husa':
        mot_tracker = SHTracker(max_age=1, min_hits=5, iou_threshold=0.3)
    elif opt.method == 'sort':
        mot_tracker = Sort(max_age=1, min_hits=5, iou_threshold=0.3)
    else:
        raise ValueError('Unknown tracking method {}'.format(opt.method))
    # s = 'M0210'
    if isinstance(opt.scene, list):
        s_list = opt.scene
    elif isinstance(opt.scene, str):
        s_list = [opt.scene]
    else:
        s_list = os.listdir(data_dir)
    for s in s_list:
        dataset = os.path.join(data_dir, s)
        if opt.save:
            # Directories
            save_dir = os.path.join(ws_path, opt.save_dir,s)  # increment run
            os.makedirs(save_dir, exist_ok=True)

        for img_name in os.listdir(dataset):
            img_path = os.path.join(dataset, img_name)
            img = cv.imread(img_path)
            # 从YOLOV5中获得当前帧的检测结果
            results = yolo(img)
            xyxy = results.xyxy[0].cpu().numpy()
            detect_boxes = xyxy[:,:4]
            detect_classes = xyxy[:,5]
            # 将检测结果放入跟踪器中，获得融合结果
            tracker = mot_tracker.update(detect_boxes, detect_classes)
            for d in tracker:
                x1 = int(float(d[0]))
                y1 = int(float(d[1]))
                x2 = int(float(d[2]))
                y2 = int(float(d[3]))
                pred_id = str(int(d[4]))
                rgb = colours[int(d[4]) % 32]
                pred_cls = d[5]
                text = pred_id + ':' + str(int(pred_cls))
                cv.rectangle(img, (x1, y1), (x2, y2), rgb, 2, 1)
                cv.putText(img, text, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 2, 1, (0, 255, 0))


            # 每一帧一展示，方便调试
            if opt.show:
                cv.namedWindow('detect_img')
                cv.imshow('img', img)
                cv.waitKey(500)

            # Save results (image with detections)
            if opt.save:
                image_name = img_path[img_path.rfind('\\') + 1:]
                save_path = os.path.join(save_dir, image_name)
                cv.imwrite(save_path, img)


if __name__ == '__main__':
    main()
