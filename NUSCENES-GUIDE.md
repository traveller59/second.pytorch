# Nuscenes Train and Eval guide

## General Tips

* Nuscenes dataset evaluation contains many hard examples, you need to modify nms parameters (decrease score threshold, increase max size). You can use ```v1.0-mini``` to tune them.

* Nuscenes dataset contain sweeps. You need to use 10 sweeps if you want to get good detection scores. Key-frame only can't get good result, so I drop support for that.

* Nuscenes dataset contains 28130 train samples and 6019 validation samples. Use Nuscenes mini train set (my custom split, ~3500 samples) when develop if you don't have 4+ GPUs. See ```NuscenesDatasetD8``` for more details.

* Some data augmentation will harm detection performance such as global rotation if their value is too large.

* Use KITTI pretrain model if possible.

## Config Guide

### Anchor Generator

1. use ```get_all_box_mean``` in nuscenes_dataset.py to get mean values of all boxes for each class.

2. change ```size``` and z-center in ```anchor_ranges``` in ```anchor_generator_range```.

3. choose thresholds, add some print function in target assigner code and train some steps to see if the threshold is too large or too small. Then tune them.

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

