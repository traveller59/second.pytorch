import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.utils.eval import get_coco_eval_result, get_official_eval_result


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, num_point_features,
                 target_assigner, feature_map_size, prep_func):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        ret = target_assigner.generate_anchors(feature_map_size)
        self._class_names = target_assigner.classes
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
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
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def ground_truth_annotations(self):
        """
        If you want to eval by my eval function, you must 
        provide this property.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use zero.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        if "annos" not in self._kitti_infos[0]:
            return None
        gt_annos = [info["annos"] for info in self._kitti_infos]
        return gt_annos

    def evaluation(self, dt_annos):
        """dt_annos have same format as ground_truth_annotations.
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None, None
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official = get_official_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        return result_official, result_coco

    def __getitem__(self, idx):
        """
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            ground_truth: {
                gt_boxes
                gt_names
                [optional]difficulty
                [optional]group_ids
            }
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata, in kitti, image index is saved in metadata
        }
        """
        info = self._kitti_infos[idx]
        kitti.convert_to_kitti_info_version2(info)
        pc_info = info["point_cloud"]
        if "points" not in pc_info:
            velo_path = pathlib.Path(pc_info['velodyne_path'])
            if not velo_path.is_absolute():
                velo_path = pathlib.Path(self._root_path) / pc_info['velodyne_path']
            velo_reduced_path = velo_path.parent.parent / (
                velo_path.parent.stem + '_reduced') / velo_path.name
            if velo_reduced_path.exists():
                velo_path = velo_reduced_path
            points = np.fromfile(
                str(velo_path), dtype=np.float32,
                count=-1).reshape([-1, self._num_point_features])
        input_dict = {
            'points': points,
        }
        if "image" in info:
            input_dict["image"] = info["image"]
        if "calib" in info:
            calib = info["calib"]
            calib_dict = {
                'rect': calib['R0_rect'],
                'Trv2c': calib['Tr_velo_to_cam'],
                'P2': calib['P2'],
            }
            input_dict["calib"] = calib_dict
        if 'annos' in info:
            annos = info['annos']
            annos_dict = {}
            # we need other objects to avoid collision when sample
            annos = kitti.remove_dontcare(annos)
            loc = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
            if "calib" in info:
                calib = info["calib"]
                gt_boxes = box_np_ops.box_camera_to_lidar(
                    gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])
                # only center format is allowed. so we need to convert
                # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
                box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
            gt_dict = {
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
            }
            if 'difficulty' in annos:
                gt_dict['difficulty'] = annos["difficulty"]
            if 'group_ids' in annos:
                gt_dict['group_ids'] = annos["group_ids"]
            input_dict["ground_truth"] = gt_dict
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {}
        if "image" in info:
            example["metadata"]["image"] = input_dict["image"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example


