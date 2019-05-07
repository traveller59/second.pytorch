import numpy as np

from second.protos import box_coder_pb2
from second.core.anchor_generator import (AnchorGeneratorStride,
                                          AnchorGeneratorRange)


def build(class_cfg):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    ag_type = class_cfg.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = class_cfg.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))
        return ag
    elif ag_type == 'anchor_generator_range':
        config = class_cfg.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))
        return ag
    elif ag_type == 'no_anchor':
        return None
    else:
        raise ValueError(" unknown anchor generator type")