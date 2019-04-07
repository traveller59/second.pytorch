# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
        if optimizer_config.fixed_weight_decay:
            optimizer_func = partial(
                torch.optim.Adam, betas=(0.9, 0.99), amsgrad=config.amsgrad)
        else:
            # regular adam
            optimizer_func = partial(
                torch.optim.Adam, amsgrad=config.amsgrad)



    # optimizer = OptimWrapper(optimizer, true_wd=optimizer_config.fixed_weight_decay, wd=config.weight_decay)
    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        wd=config.weight_decay,
        true_wd=optimizer_config.fixed_weight_decay,
        bn_wd=True)
    print(hasattr(optimizer, "_amp_stash"), '_amp_stash')
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
