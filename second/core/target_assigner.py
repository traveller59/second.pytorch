import numpy as np
from collections import OrderedDict

from second.core import box_np_ops, region_similarity
from second.core.target_ops import create_target_np
from second.utils.timer import simple_timer


class TargetAssigner:
    def __init__(self,
                 box_coder,
                 anchor_generators,
                 classes,
                 feature_map_sizes,
                 positive_fraction=None,
                 region_similarity_calculators=None,
                 sample_size=512,
                 assign_per_class=True):
        self._box_coder = box_coder
        self._anchor_generators = anchor_generators
        self._sim_calcs = region_similarity_calculators
        box_ndims = [a.ndim for a in anchor_generators]
        assert all([e == box_ndims[0] for e in box_ndims])
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size
        self._classes = classes
        self._assign_per_class = assign_per_class
        self._feature_map_sizes = feature_map_sizes

    @property
    def box_coder(self):
        return self._box_coder

    @property
    def classes(self):
        return self._classes

    def assign(self,
               anchors,
               anchors_dict,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               gt_names=None,
               matched_thresholds=None,
               unmatched_thresholds=None,
               importance=None):
        if self._assign_per_class:
            return self.assign_per_class(anchors_dict, gt_boxes, anchors_mask,
                                         gt_classes, gt_names, importance=importance)
        else:
            return self.assign_all(anchors, gt_boxes, anchors_mask, gt_classes,
                                   matched_thresholds, unmatched_thresholds, importance=importance)

    def assign_all(self,
                   anchors,
                   gt_boxes,
                   anchors_mask=None,
                   gt_classes=None,
                   matched_thresholds=None,
                   unmatched_thresholds=None,
                   importance=None):
        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
        else:
            prune_anchor_fn = None

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._sim_calcs[0].compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)
        return create_target_np(
            anchors,
            gt_boxes,
            similarity_fn,
            box_encoding_fn,
            prune_anchor_fn=prune_anchor_fn,
            gt_classes=gt_classes,
            matched_threshold=matched_thresholds,
            unmatched_threshold=unmatched_thresholds,
            positive_fraction=self._positive_fraction,
            rpn_batch_size=self._sample_size,
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size,
            gt_importance=importance)

    def assign_per_class(self,
                         anchors_dict,
                         gt_boxes,
                         anchors_mask=None,
                         gt_classes=None,
                         gt_names=None,
                         importance=None):
        """this function assign target individally for each class.
        recommend for multi-class network.
        """

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        anchor_loc_idx = 0
        anchor_gene_idx = 0
        for class_name, anchor_dict in anchors_dict.items():

            def similarity_fn(anchors, gt_boxes):
                anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
                gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
                return self._sim_calcs[anchor_gene_idx].compare(
                    anchors_rbv, gt_boxes_rbv)

            mask = np.array([c == class_name for c in gt_names],
                            dtype=np.bool_)
            feature_map_size = anchor_dict["anchors"].shape[:3]
            num_loc = anchor_dict["anchors"].shape[-2]
            if anchors_mask is not None:
                anchors_mask = anchors_mask.reshape(-1)
                a_range = self.anchors_range(class_name)
                anchors_mask_class = anchors_mask[a_range[0]:a_range[1]].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None
            # print(f"num of {class_name}:", np.sum(mask))
            targets = create_target_np(
                anchor_dict["anchors"].reshape(-1, self.box_ndim),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict["matched_thresholds"],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,
                rpn_batch_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
                gt_importance=importance)
            # print(f"num of positive:", np.sum(targets["labels"] == self.classes.index(class_name) + 1))
            anchor_loc_idx += num_loc
            targets_list.append(targets)
            anchor_gene_idx += 1

        targets_dict = {
            "labels": [t["labels"] for t in targets_list],
            "bbox_targets": [t["bbox_targets"] for t in targets_list],
            "importance": [t["importance"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = np.concatenate([
            v.reshape(-1, self.box_coder.code_size)
            for v in targets_dict["bbox_targets"]
        ],
                                                      axis=0)
        targets_dict["bbox_targets"] = targets_dict["bbox_targets"].reshape(
            -1, self.box_coder.code_size)
        targets_dict["labels"] = np.concatenate(
            [v.reshape(-1) for v in targets_dict["labels"]],
            axis=0)
        targets_dict["importance"] = np.concatenate(
            [v.reshape(-1) for v in targets_dict["importance"]],
            axis=0)
        targets_dict["labels"] = targets_dict["labels"].reshape(-1)
        targets_dict["importance"] = targets_dict["importance"].reshape(-1)

        return targets_dict

    def generate_anchors(self, feature_map_size):
        anchors_list = []
        ndim = len(feature_map_size)
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        match_list, unmatch_list = [], []
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes
        else:
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        idx = 0
        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            anchors_list.append(anchors.reshape(-1, self.box_ndim))
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
            idx += 1
        anchors = np.concatenate(anchors_list, axis=0)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }

    def generate_anchors_dict(self, feature_map_size):
        ndim = len(feature_map_size)
        anchors_list = []
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        match_list, unmatch_list = [], []
        anchors_dict = OrderedDict()
        for a in self._anchor_generators:
            anchors_dict[a.class_name] = {}
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes
        else:
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        idx = 0
        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size

            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]["anchors"] = anchors.reshape(-1, self.box_ndim)
            anchors_dict[class_name]["matched_thresholds"] = match_list[-1]
            anchors_dict[class_name]["unmatched_thresholds"] = unmatch_list[-1]
            idx += 1
        return anchors_dict

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    @property
    def box_ndim(self):
        return self._anchor_generators[0].ndim

    def num_anchors(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        class_idx = self._classes.index(class_name)
        ag = self._anchor_generators[class_idx]
        feature_map_size = self._feature_map_sizes[class_idx]
        return np.prod(feature_map_size) * ag.num_anchors_per_localization

    def anchors_range(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        num_anchors = 0
        anchor_ranges = []
        for name in self._classes:
            anchor_ranges.append((num_anchors, num_anchors + self.num_anchors(name)))
            num_anchors += anchor_ranges[-1][1] - num_anchors
        return anchor_ranges[self._classes.index(class_name)]
        
    def num_anchors_per_location_class(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        class_idx = self._classes.index(class_name)
        return self._anchor_generators[class_idx].num_anchors_per_localization