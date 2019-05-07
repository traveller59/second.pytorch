# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

from second.protos import input_reader_pb2
from second.data.dataset import get_dataset_class
from second.data.preprocess import prep_pointcloud
from second.core import box_np_ops
import numpy as np
from second.builder import dbsampler_builder
from functools import partial
from second.utils.config_tool import get_downsample_factor

def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner,
          multi_gpu=False):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    prep_cfg = input_reader_config.preprocess
    dataset_cfg = input_reader_config.dataset
    num_point_features = model_config.num_point_features
    out_size_factor = get_downsample_factor(model_config)
    assert out_size_factor > 0
    cfg = input_reader_config
    db_sampler_cfg = prep_cfg.database_sampler
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0 or db_sampler_cfg.database_info_path != "":  # enable sample
        db_sampler = dbsampler_builder.build(db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    print("feature_map_size", feature_map_size)
    assert all([n != '' for n in target_assigner.classes]), "you must specify class_name in anchor_generators."
    dataset_cls = get_dataset_class(dataset_cfg.dataset_class_name)
    assert dataset_cls.NumPointFeatures >= 3, "you must set this to correct value"
    assert dataset_cls.NumPointFeatures == num_point_features, "currently you need keep them same"
    prep_func = partial(
        prep_pointcloud,
        root_path=dataset_cfg.kitti_root_path,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=prep_cfg.max_number_of_voxels,
        remove_outside_points=False,
        remove_unknown=prep_cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=prep_cfg.shuffle_points,
        gt_rotation_noise=list(prep_cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(prep_cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(prep_cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(prep_cfg.global_scaling_uniform_noise),
        global_random_rot_range=list(
            prep_cfg.global_random_rotation_range_per_object),
        global_translate_noise_std=list(prep_cfg.global_translate_noise_std),
        db_sampler=db_sampler,
        num_point_features=dataset_cls.NumPointFeatures,
        anchor_area_threshold=prep_cfg.anchor_area_threshold,
        gt_points_drop=prep_cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=prep_cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=prep_cfg.remove_points_after_sample,
        remove_environment=prep_cfg.remove_environment,
        use_group_id=prep_cfg.use_group_id,
        out_size_factor=out_size_factor,
        multi_gpu=multi_gpu,
        min_points_in_gt=prep_cfg.min_num_of_points_in_gt,
        random_flip_x=prep_cfg.random_flip_x,
        random_flip_y=prep_cfg.random_flip_y,
        sample_importance=prep_cfg.sample_importance)

    ret = target_assigner.generate_anchors(feature_map_size)
    class_names = target_assigner.classes
    anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
    anchors_list = []
    for k, v in anchors_dict.items():
        anchors_list.append(v["anchors"])
    
    # anchors = ret["anchors"]
    anchors = np.concatenate(anchors_list, axis=0)
    anchors = anchors.reshape([-1, target_assigner.box_ndim])
    assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
        "anchors_dict": anchors_dict,
    }
    prep_func = partial(prep_func, anchor_cache=anchor_cache)
    dataset = dataset_cls(
        info_path=dataset_cfg.kitti_info_path,
        root_path=dataset_cfg.kitti_root_path,
        class_names=class_names,
        prep_func=prep_func)

    return dataset
