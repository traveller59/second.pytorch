
#include "point_cloud_ops.h"

PYBIND11_MODULE(point_cloud_ops_cc, m) {
  m.doc() = "pybind11 example plugin";
  m.def("points_to_voxel", &points_to_voxel<float, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
        "num_points_per_voxel"_a = 4, "voxel_size"_a = 5, "coors_range"_a = 6,
        "max_points"_a = 7, "max_voxels"_a = 8);
  m.def("points_to_voxel", &points_to_voxel<double, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
        "num_points_per_voxel"_a = 4, "voxel_size"_a = 5, "coors_range"_a = 6,
        "max_points"_a = 7, "max_voxels"_a = 8);
}