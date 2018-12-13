# SECOND for KITTI object detection
SECOND detector. Based on my unofficial implementation of VoxelNet with some improvements.

ONLY support python 3.6+, pytorch 0.4.1+. Don't support pytorch 0.4.0. Tested in Ubuntu 16.04/18.04.

* Ubuntu 18.04 have speed problem in my environment and may can't build/usr SparseConvNet.

### Performance in KITTI validation set (50/50 split, people have problems, need to be tuned.)

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.80, 88.97, 87.52
bev  AP:89.96, 86.69, 86.11
3d   AP:87.43, 76.48, 74.66
aos  AP:90.68, 88.39, 86.57
Car AP@0.70, 0.50, 0.50:
bbox AP:90.80, 88.97, 87.52
bev  AP:90.85, 90.02, 89.36
3d   AP:90.85, 89.86, 89.05
aos  AP:90.68, 88.39, 86.57
```

## Install

### 1. Clone code

```bash
git clone https://github.com/traveller59/second.pytorch.git
cd ./second.pytorch/second
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```

If you don't have Anaconda:

```bash
pip install numba
```

Follow instructions in https://github.com/facebookresearch/SparseConvNet to install SparseConvNet.

Install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```


### 3. Setup cuda for numba

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. add second.pytorch/ to PYTHONPATH

## Prepare dataset

* Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

* Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

* Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

* Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```

## Usage

### train

```bash
python ./pytorch/train.py train --config_path=./configs/car.config --model_dir=/path/to/model_dir
```

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=3 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 40 epoch to reach 74 AP in car moderate 3D in Kitti validation dateset.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/path/to/model_dir
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### pretrained model

Before using pretrained model, you need to modify some file in SparseConvNet because the pretrained model doesn't support SparseConvNet master:

* convolution.py
```Python
# self.weight = Parameter(torch.Tensor(
#     self.filter_volume, nIn, nOut).normal_(
#     0,
#     std))
self.weight = Parameter(torch.Tensor(
    self.filter_volume * nIn, nOut).normal_(
    0,
    std))
# ...
# output.features = ConvolutionFunction.apply(
#     input.features,
#     self.weight,
output.features = ConvolutionFunction.apply(
    input.features,
    self.weight.view(self.filter_volume, self.nIn, self.nOut),
```

* submanifoldConvolution.py
```Python
# self.weight = Parameter(torch.Tensor(
#     self.filter_volume, nIn, nOut).normal_(
#     0,
#     std))
self.weight = Parameter(torch.Tensor(
    self.filter_volume * nIn, nOut).normal_(
    0,
    std))
# ...
# output.features = SubmanifoldConvolutionFunction.apply(
#     input.features,
#     self.weight,
output.features = SubmanifoldConvolutionFunction.apply(
    input.features,
    self.weight.view(self.filter_volume, self.nIn, self.nOut),
```

You can download pretrained models in [google drive](https://drive.google.com/open?id=1eblyuILwbxkJXfIP5QlALW5N_x5xJZhL). The car model is corresponding to car.config, the car_tiny model is corresponding to car.tiny.config and the people model is corresponding to people.config.

## Docker

You can use a prebuilt docker for testing:
```
docker pull scrin/second-pytorch 
```
Then run:
```
nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host second-pytorch:latest
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/root/model/car
...
```

Currently there is a problem that training and evaluating in docker is very slow.

## Try Kitti Viewer Web

### Major step

1. run ```python ./kittiviewer/backend.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/viewerweb.png)



## Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/simpleguide.png)

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
