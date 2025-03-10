# vision-bf

This is the repository for reproducing the paper [Vision-Assisted 3-D Predictive Beamforming
for Green UAV-to-Vehicle Communications](https://ieeexplore.ieee.org/abstract/document/10008064).



This project is implemented based on [pytorch](https://pytorch.org/). A fundamental tutorial can be found either on their official website or [here](https://www.learnpytorch.io/?continueFlag=a8c4e27c1f2d0ca982de9ae3592b2ba4).

## Get Started


- Create your `conda` environment named `vision-bf` with `python 3.7` and `pip` installed
  ```
  conda create -n vision-bf python=3.7 pip
  ```

- Activate the created environment and install the packages required by `yolov5` and this repo
  ```
  conda activate vision-bf
  pip install -e .
  pip install -r ./yolov5/requirements.txt
  ```
  If you have CUDA installed, please install the CUDA version pytorch
  ```
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```

- Download the `UAVDT` dataset from [here](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5), and you should have two `zip` files, `UAV-benchmark-M` and `UAV-benchmark-MOTD_v1.0.zip`. Unzip them into the folder `./data`.

- To generate dataset that can be used for training `yolov5`, run the script `./scripts/generate_dataset.py`
  ```
  python ./scripts/generate_dataset.py
  ```

- To train the `yolov5` model on the generated dataset, copy the configuration files under `vision-bf/configs` into the corresponding locations of `yolov5` repo
  ```
  cp ./configs/uavdt.yaml ./yolov5/data/
  cp ./configs/hyp.uavdt.yaml ./yolov5/data/hyps/
  ```
  and start the training
  ```
  cd ./yolov5
  python train.py --data uavdt.yaml --hyp hyp.uavdt.yaml --batch-size 32 --epoch 100 --workers 1
  ```
  If you train the model on computer with GPUs, for examples, 2 GPUs and a 64-core CPU, to make use of the computing resource, you can run
  ```
  python train.py --data uavdt.yaml --hyp hyp.uavdt.yaml --batch-size 32 --epoch 100 --workers 64 --device 0,1
  ```
  See [here](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) for more information on training.

- After training, the results and trained model can be found under `yolov5/runs/`. To test the trained model on the test dataset, use the provided script `yolov5/val.py` and specify the model path. We have already trained a model and copied the results to `vision-bf/runs/`. To test on the provided trained model, run 
  ```
  python val.py --data uavdt.yaml --weights ../runs/train/exp/weights/best.pt
  ```
  and the results will be printed onto the screen. For example,
  ```
  Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 127/127
  all       4041      80258       0.98      0.956      0.985      0.755
  Speed: 0.1ms pre-process, 2.6ms inference, 0.6ms NMS per image at shape (32, 3, 640, 640)
  Results saved to runs/val/exp2
  ```

## Tracking

The tracking algorithms are implemented in `src/sort` and `src/sage_husa` based on the repos [yolov5_sort](https://gitcode.net/lzzzzzzm/yolov5_sort/-/tree/master) and [sort](https://github.com/abewley/sort). To see the tracking performance, run 
```
python scripts/tracking.py
```
To see the performance of `sort`, run
```
python scripts/tracking.py --method sort
```


## Folder Structure

```
├───configs      --- configuration files
├───data         --- folder containing dataset
├───notebook     --- all jupyter notebooks stored here
├───scripts      --- all python scripts stored here
├───src          --- folder containing all functions, classes
└───tests        --- folder for tests
```

## Resources

Related papers, websites and other important resources go here.

### Papers

### Websites

- [UAVDT Dataset](https://paperswithcode.com/dataset/uavdt)
- [yolov5](https://github.com/ultralytics/yolov5)#   M i m o 
 
 
