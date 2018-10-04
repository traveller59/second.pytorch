from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from second.core import box_np_ops
import numpy as np

class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 8 if self.vec_encode else 7

    def _encode(self, boxes, anchors):
        return box_np_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        return box_np_ops.second_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)


class BevBoxCoder(BoxCoder):
    """WARNING: this coder will return encoding with size=5, but 
    takes size=7 boxes, anchors
    """
    def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0):
        super().__init__()
        self.linear_dim = linear_dim
        self.z_fixed = z_fixed
        self.h_fixed = h_fixed
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 6 if self.vec_encode else 5

    def _encode(self, boxes, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        boxes = boxes[..., [0, 1, 3, 4, 6]]
        return box_np_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_np_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
        z_fixed = np.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
        h_fixed = np.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
        return np.concatenate([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], axis=-1)



