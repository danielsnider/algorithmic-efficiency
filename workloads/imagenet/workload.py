import spec


class ImagenetWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    return 0.76

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return [0.485 * 255, 0.456 * 255, 0.406 * 255]

  @property
  def train_stddev(self):
    return [0.229 * 255, 0.224 * 255, 0.225 * 255]

  @property
  def max_allowed_runtime_sec(self):
    if 'imagenet2012' in self.dataset:
      return 111600 # 31 hours
    if 'imagenette' == self.dataset:
      return 3600 # 60 minutes

  @property
  def eval_period_time_sec(self):
    if 'imagenet2012' in self.dataset:
      return 6000 # 100 mins
    if 'imagenette' == self.dataset:
      return 30 # 30 seconds

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError

  def build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))
