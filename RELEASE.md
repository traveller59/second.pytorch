# Release 1.5

## Major Features and Improvements

1. New sparse convolution based models. VFE-based old models are deprecated. Now the model looks like this:
points([N, 4])->voxels([N, 5, 4])->Features([N, 4])->Sparse Convolution Networks->RPN. See [this](https://github.com/traveller59/second.pytorch/blob/master/second/pytorch/models/middle.py) for more details of sparse conv networks.
2. The [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) is deprecated. New library [spconv](https://github.com/traveller59/spconv) is introduced.
3. Super converge (from fastai) is implemented. Now all network can converge to a good result with only 50~80 epoch. For example. ```car.fhd.config``` only needs 50 epochs to reach 78.3 AP (car mod 3d).
4. Target assigner now works correctly when using multi-class.