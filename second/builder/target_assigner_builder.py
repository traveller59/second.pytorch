import numpy as np

from second.core.target_assigner import TargetAssigner
from second.protos import target_pb2, anchors_pb2
from second.builder import similarity_calculator_builder
from second.builder import anchor_generator_builder

def build(target_assigner_config, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(target_assigner_config, (target_pb2.TargetAssigner)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    classes_cfg = target_assigner_config.class_settings
    anchor_generators = []
    classes = []
    feature_map_sizes = []
    for class_setting in classes_cfg:
        anchor_generator = anchor_generator_builder.build(class_setting)
        if anchor_generator is not None:
            anchor_generators.append(anchor_generator)
        else:
            assert target_assigner_config.assign_per_class is False
        classes.append(class_setting.class_name)
        feature_map_sizes.append(class_setting.feature_map_size)
    similarity_calcs = []
    for class_setting in classes_cfg:
        similarity_calcs.append(similarity_calculator_builder.build(
            class_setting.region_similarity_calculator))

    positive_fraction = target_assigner_config.sample_positive_fraction
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        feature_map_sizes=feature_map_sizes,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config.sample_size,
        region_similarity_calculators=similarity_calcs,
        classes=classes,
        assign_per_class=target_assigner_config.assign_per_class)
    return target_assigner
