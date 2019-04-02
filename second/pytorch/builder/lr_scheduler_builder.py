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

from torchplus.train import learning_schedules_fastai as lsf
import torch

def build(optimizer_config, optimizer, total_step):
  """Create lr scheduler based on config. note that
  lr_scheduler must accept a optimizer that has been restored.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')

  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, total_step=total_step)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, total_step=total_step)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, total_step=total_step)

  return lr_scheduler

def _create_learning_rate_scheduler(learning_rate_config, optimizer, total_step):
  """Create optimizer learning rate scheduler based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  lr_scheduler = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'multi_phase':
    config = learning_rate_config.multi_phase
    lr_phases = []
    mom_phases = []
    for phase_cfg in config.phases:
      lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
      mom_phases.append((phase_cfg.start, phase_cfg.momentum_lambda_func))
    lr_scheduler = lsf.LRSchedulerStep(
      optimizer,total_step, lr_phases, mom_phases)

  if learning_rate_type == 'one_cycle':
    config = learning_rate_config.one_cycle
    lr_scheduler = lsf.OneCycle(
      optimizer, total_step, config.lr_max, list(config.moms), config.div_factor, config.pct_start)
  if learning_rate_type == 'exponential_decay':
    config = learning_rate_config.exponential_decay
    lr_scheduler = lsf.ExponentialDecay(
      optimizer, total_step, config.initial_learning_rate, config.decay_length, config.decay_factor, config.staircase)
  if learning_rate_type == 'manual_stepping':
    config = learning_rate_config.manual_stepping
    lr_scheduler = lsf.ManualStepping(
      optimizer, total_step, list(config.boundaries), list(config.rates))

  if lr_scheduler is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return lr_scheduler