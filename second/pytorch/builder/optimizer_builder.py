"""Functions to build DetectionModel training optimizers."""

from torchplus.train import learning_schedules
from torchplus.train import optim
import torch
from torch import nn
from torchplus.train.fastai_optim import OptimWrapper, FastAIMixedOptim
from functools import partial


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))

flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]

get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]


def build(optimizer_config, net, name=None, mixed=False, loss_scale=512.0):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer_func = partial(
            torch.optim.RMSprop,
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer_func = partial(
            torch.optim.SGD,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=config.amsgrad)

    # optimizer = OptimWrapper(optimizer, true_wd=optimizer_config.fixed_weight_decay, wd=config.weight_decay)
    if mixed:
        optimizer = FastAIMixedOptim.create(
            optimizer_func,
            3e-3,
            get_layer_groups(net),
            net,
            loss_scale=loss_scale,
            wd=config.weight_decay,
            true_wd=optimizer_config.fixed_weight_decay,
            bn_wd=True)
    else:
        optimizer = OptimWrapper.create(
            optimizer_func,
            3e-3,
            get_layer_groups(net),
            wd=config.weight_decay,
            true_wd=optimizer_config.fixed_weight_decay,
            bn_wd=True)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        raise ValueError('torch don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_type
    else:
        optimizer.name = name
    return optimizer
