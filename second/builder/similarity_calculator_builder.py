import numpy as np

from second.core import region_similarity
from second.protos import similarity_pb2


def build(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    similarity_type = similarity_config.WhichOneof('region_similarity')
    if similarity_type == 'rotate_iou_similarity':
        return region_similarity.RotateIouSimilarity()
    elif similarity_type == 'nearest_iou_similarity':
        return region_similarity.NearestIouSimilarity()
    elif similarity_type == 'distance_similarity':
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha)
    else:
        raise ValueError("unknown similarity type")
