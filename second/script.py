from second.pytorch.train import train, evaluate
from google.protobuf import text_format
from second.protos import pipeline_pb2
from pathlib import Path
from second.utils import config_tool


def train_multi_rpn_layer_num():
    config_path = "./configs/car.lite.config"
    model_root = Path.home() / "second_test"  # don't forget to change this.
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    layer_nums = [2, 4, 7, 9]
    for l in layer_nums:
        model_dir = str(model_root / f"car_lite_L{l}")
        model_cfg.rpn.layer_nums[:] = [l]
        train(config, model_dir)


def eval_multi_threshold():
    config_path = "./configs/car.fhd.config"
    ckpt_name = "/path/to/your/model_ckpt" # don't forget to change this.
    assert "/path/to/your" not in ckpt_name
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    threshs = [0.3]
    for thresh in threshs:
        model_cfg.nms_score_threshold = thresh
        # don't forget to change this.
        result_path = Path.home() / f"second_test_eval_{thresh:.2f}"
        evaluate(
            config,
            result_path=result_path,
            ckpt_path=str(ckpt_name),
            batch_size=1,
            measure_time=True)


if __name__ == "__main__":
    eval_multi_threshold()