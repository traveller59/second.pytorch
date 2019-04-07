# Release 1.6.0alpha

## Major Features and Improvements
1. New dataset API (unstable during alpha), almost completely remove kitti-specific code. you can add your custom dataset by following steps: 
(1): implement all Dataset API functions
(2): use web visualization tool to check whether the box is correct.
(3): add your dataset to all_dataset.py, change the dataset_class_name in config file.

2. Add [NuScenes](https://www.nuscenes.org) dataset support (incomplete in 1.6.0alpha), I plan to reproduce the NDS score in their paper.

3. Add [pointpillars](https://github.com/nutonomy/second.pytorch) to this repo.

4. Full Tensorboard support.

5. FP16 and multi-gpu (need test, I only have one gpu) support.

## Minor Improvements and Bug fixes

1. Move all data-specific functions to their corresponding dataset file.

2. Improved config file structure, remove some unused item.

3. remove much unused and deprecated code.

4. add two learning rate scheduler: exp decay and manual step

# Release 1.5.1

## Minor Improvements and Bug fixes

1. Better support for custom lidar data. You need to check KittiDataset for more details. (no test yet, I don't have custom data)
* Change all box to center format. 
* Change kitti info format, now you need to regenerate kitti infos and gt database.
* Eval functions now support custom data evaluation. you need to specify z_center and z_axis in eval function.
2. Better RPN, you can add custom block by inherit RPNBase and implement _make_layer method.
3. Update pretrained model.
4. Add a simple inference notebook. everyone should start this project by that notebook.
5. Add windows support. Training on windows is slow than linux.

# Release 1.5

## Major Features and Improvements

1. New sparse convolution based models. VFE-based old models are deprecated. Now the model looks like this:
points([N, 4])->voxels([N, 5, 4])->Features([N, 4])->Sparse Convolution Networks->RPN. See [this](https://github.com/traveller59/second.pytorch/blob/master/second/pytorch/models/middle.py) for more details of sparse conv networks.
2. The [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) is deprecated. New library [spconv](https://github.com/traveller59/spconv) is introduced.
3. Super converge (from fastai) is implemented. Now all network can converge to a good result with only 50~80 epoch. For example. ```car.fhd.config``` only needs 50 epochs to reach 78.3 AP (car mod 3d).
4. Target assigner now works correctly when using multi-class.

