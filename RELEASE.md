# Release 1.5

## Major Features and Improvements

*   New sparse convolution based models. VFE-based old models are deprecated.
*   Super converge (fastai) is implemented. Now all network can converge to 
    a good result with only 50~80 epoch. For example. ```car.fhd.config``` only needs 50 epochs to reach 78.3 AP (car mod 3d).
*   Target assigner now works correctly when using multi-class.