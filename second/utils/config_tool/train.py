from second.protos.optimizer_pb2 import Optimizer, LearningRate, OneCycle, ManualStepping, ExponentialDecay
from second.protos.sampler_pb2 import Sampler
from second.utils.config_tool import read_config
from pathlib import Path
from google.protobuf import text_format
from second.data.all_dataset import get_dataset_class

def _get_optim_cfg(train_config, optim):
    if optim == "adam_optimizer":
        return train_config.optimizer.adam_optimizer
    elif optim == "rms_prop_optimizer":
        return train_config.optimizer.rms_prop_optimizer
    elif optim == "momentum_optimizer":
        return train_config.optimizer.momentum_optimizer
    else:
        raise NotImplementedError


def manual_stepping(train_config, boundaries, rates, optim="adam_optimizer"):
    optim_cfg = _get_optim_cfg(train_config, optim)
    optim_cfg.learning_rate.manual_stepping.CopyFrom(
        ManualStepping(boundaries=boundaries, rates=rates))


def exp_decay(train_config,
              init_lr,
              decay_length,
              decay_factor,
              staircase=True,
              optim="adam_optimizer"):
    optim_cfg = _get_optim_cfg(train_config, optim)
    optim_cfg.learning_rate.exponential_decay.CopyFrom(
        ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_length=decay_length,
            decay_factor=decay_factor,
            staircase=staircase))


def one_cycle(train_config,
              lr_max,
              moms,
              div_factor,
              pct_start,
              optim="adam_optimizer"):
    optim_cfg = _get_optim_cfg(train_config, optim)
    optim_cfg.learning_rate.one_cycle.CopyFrom(
        OneCycle(
            lr_max=lr_max,
            moms=moms,
            div_factor=div_factor,
            pct_start=pct_start))

def _div_up(a, b):
    return (a + b - 1) // b

def set_train_step(config,
                    epochs,
                    eval_epoch):
    input_cfg = config.train_input_reader
    train_cfg = config.train_config
    batch_size = input_cfg.batch_size
    dataset_name = input_cfg.dataset.dataset_class_name
    ds = get_dataset_class(dataset_name)(
        root_path=input_cfg.dataset.kitti_root_path,
        info_path=input_cfg.dataset.kitti_info_path,
    )
    num_examples_after_sample = len(ds)
    step_per_epoch = _div_up(num_examples_after_sample, batch_size)
    step_per_eval = step_per_epoch * eval_epoch
    total_step = step_per_epoch * epochs
    train_cfg.steps = total_step
    train_cfg.steps_per_eval = step_per_eval

def disable_sample(config):
    input_cfg = config.train_input_reader
    input_cfg.database_sampler.CopyFrom(Sampler())

def disable_per_gt_aug(config):
    prep_cfg = config.train_input_reader.preprocess
    prep_cfg.groundtruth_localization_noise_std[:] = [0, 0, 0]
    prep_cfg.groundtruth_rotation_uniform_noise[:] = [0, 0]

def disable_global_aug(config):
    prep_cfg = config.train_input_reader.preprocess
    prep_cfg.global_rotation_uniform_noise[:] = [0, 0]
    prep_cfg.global_scaling_uniform_noise[:] = [0, 0]
    prep_cfg.global_random_rotation_range_per_object[:] = [0, 0]
    prep_cfg.global_translate_noise_std[:] = [0, 0, 0]

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[2] / "configs/car.lite.config"
    config = read_config(path)
    manual_stepping(config.train_config, [0.8, 0.9], [1e-4, 1e-5, 1e-6])
    
    print(text_format.MessageToString(config, indent=2))