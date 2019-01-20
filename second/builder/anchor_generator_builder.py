import numpy as np

from second.protos import box_coder_pb2
from second.core.anchor_generator import (
    AnchorGeneratorStride, AnchorGeneratorRange)


def build(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    ag_type = anchor_config.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = anchor_config.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_name=config.class_name)
        return ag
    elif ag_type == 'anchor_generator_range':
        config = anchor_config.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_name=config.class_name)
        return ag
    else:
        raise ValueError(" unknown anchor generator type")