# Nuscenes Train and Eval guide

## General Tips

* Nuscenes dataset evaluation contains many hard examples, you need to modify nms parameters (decrease score threshold, increase max size). You can use ```v1.0-mini``` to tune them.

* Nuscenes dataset contain sweeps. You need to use 10 sweeps if you want to get good detection scores. Key-frame only can't get good result, so I drop support for that.

* Nuscenes dataset contains 28130 train samples and 6019 validation samples. Use Nuscenes mini train set (my custom split, ~3500 samples) when develop if you don't have 4+ GPUs. See ```NuscenesDatasetD8``` for more details.

* Some data augmentation will harm detection performance such as global rotation if their value is too large.

* Use KITTI pretrain model if possible. You can use a pointpillars xyres_16 car model in [google drive](https://drive.google.com/open?id=1YOpgRkBgmSAJwMknoXmitEArNitZz63C) as pretrained model.

* Multi-class NMS can increase performance of large objects. The parameters of mcnms should be grid-searched, don't enable mcmns during develop.

* Database sample can greatly increase performance of bus. Currently no effect on other class.

## Config Guide

### Anchor Generator

1. use ```get_all_box_mean``` in nuscenes_dataset.py to get mean values of all boxes for each class.

2. change ```size``` and z-center in ```anchor_ranges``` in ```anchor_generator_range```.

3. choose thresholds: use ```helper_tune_target_assigner``` to get instance count and assigned anchors count. Then tune them.

4. add ```region_similarity_calculator```. If your anchors are too sparse, you need to use ```distance_similarity``` instead of ```nearest_iou_similarity``` for small classes such as pedestrian.

5. If you want to train with velocity, add ```custom_values``` to anchor generator. you can add two zeros. After that, anchors' shape will become ```[N, 9]```.

### Preprocess

1. disable all ground-truth noise.

2. ```global_rotation_uniform_noise``` may decrease performance.

3. disable ```database_sampler``` by delete all content in ```database_sampler```.

### Train

Use ```set_train_step``` in utils.config_tool.train if you don't want to calculate them manually.

## Develop Guide

* uncomment vis functions in prep_pointcloud to see assigned anchors and point cloud after data augmentation to ensure no bug in preprocess.

* use code such as code in script_server.py instead of use commands in terminal.

## Config reference

* all.fhd.config

1. Block filtering is enabled by default to decrease number of voxels.

2. Use ```upsample_strides: [0.5]``` to decrease number of anchors and increase training speed.

* all.pp.mhead.config

This config use custom ```VoxelNet``` class.

1. Use ```VoxelNetNuscenesMultiHead``` and write a multi-head network to handle small objects.

2. Output of multi head network MUST match order of class settings in config file.

3. If you use custom VoxelNet class, you MUST set feature_map_size for every class in class settings.

* all.pp.largea.config

This config use only large anchors.

1. you can use ```no_anchor``` to disable anchor generation of a class.

2. you must set ```assign_per_class``` to false when use ```no_anchor```.


## Reference Performance (Single GPU)

* all.pp.lowa.config: 30 epoch, 1/2 dataset (NuscenesDatasetD2), train speed: 12 sample/s, ~50000 anchors

```
car Nusc dist AP@0.5, 1.0, 2.0, 4.0
58.85, 76.12, 80.65, 82.49
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
2.55, 15.42, 27.19, 32.03
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.02, 0.31
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
8.61, 14.30, 15.00, 15.53
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
39.14, 49.29, 53.50, 57.03
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
12.58, 18.92, 22.79, 27.99
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 1.10, 7.42, 20.91
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
5.44, 15.78, 22.77, 27.05
barrier Nusc dist AP@0.5, 1.0, 2.0, 4.0
7.54, 34.54, 44.52, 49.80
```

* all.pp.mida.config: 30 epoch, 1/2 dataset (NuscenesDatasetD2), train speed: 10 sample/s, ~200000 anchors

car Nusc dist AP@0.5, 1.0, 2.0, 4.0
61.91, 76.75, 80.94, 82.53
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
8.72, 25.76, 39.13, 42.55
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.27, 1.18
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
12.62, 17.82, 18.31, 18.82
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
55.50, 58.74, 61.27, 63.93
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
17.47, 20.30, 23.56, 28.63
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 4.76, 18.82, 28.26
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
8.39, 20.62, 27.37, 31.12
barrier Nusc dist AP@0.5, 1.0, 2.0, 4.0
10.61, 31.31, 40.60, 46.33

* all.pp.config: 50 epoch, 1/8 dataset (NuscenesDatasetD8), train speed: 4 sample/s, ~1200000 anchors

```
car Nusc dist AP@0.5, 1.0, 2.0, 4.0
62.90, 73.07, 76.77, 78.79
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.53, 26.17, 38.01, 40.60
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.44, 1.43
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.25, 12.90, 13.69, 14.11
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
61.44, 62.61, 64.09, 66.35
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
11.63, 13.14, 15.81, 21.22
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.80, 9.90, 17.61, 23.26
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.81, 21.40, 27.55, 30.34
```

* all.fhd.config: train speed: 5.5 sample/s. comming soon