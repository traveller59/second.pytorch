import numpy as np
from second.core import box_np_ops

class AnchorGenerator:
    @property
    def class_name(self):
        raise NotImplementedError

    @property
    def num_anchors_per_localization(self):
        raise NotImplementedError

    def generate(self, feature_map_size):
        raise NotImplementedError

    @property 
    def ndim(self):
        raise NotImplementedError


class AnchorGeneratorStride(AnchorGenerator):
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 class_name=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 custom_values=(),
                 dtype=np.float32):
        super().__init__()
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def class_name(self):
        return self._class_name

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        res = box_np_ops.create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._dtype)
        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    @property 
    def ndim(self):
        return 7 + len(self._custom_values)

    @property 
    def custom_ndim(self):
        return len(self._custom_values)

class AnchorGeneratorRange(AnchorGenerator):
    def __init__(self,
                 anchor_ranges,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 class_name=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 custom_values=(),
                 dtype=np.float32):
        super().__init__()
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def class_name(self):
        return self._class_name

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        res = box_np_ops.create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)

        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    @property 
    def ndim(self):
        return 7 + len(self._custom_values)

    @property 
    def custom_ndim(self):
        return len(self._custom_values)