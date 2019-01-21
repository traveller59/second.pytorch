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

  if lr_scheduler is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return lr_scheduler