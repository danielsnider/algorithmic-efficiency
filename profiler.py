
nvtx_enabled = False
torchprofiler_enabled = False

def noop():
  pass

def nvtx_start():
  import torch
  torch.cuda.cudart().cudaProfilerStart()

def nvtx_stop():
  import torch
  torch.cuda.cudart().cudaProfilerStop()

class no_op_nvtx:
  class annotate:
    def __init__(self, *args, **kargs):
      pass

    def __enter__(self):
      pass
      return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
      pass


def torch_profiler():
  import torch
  torch.cuda.cudart().cudaProfilerStart()
  prof = torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        # record_shapes=True,
        # profile_memory=True, # makes it slower
        # with_stack=True
  )
  return TorchProfiler

def no_op_torch_profiler():
  class TorchProfiler:
    def step(self):
      pass
    def start(self):
      pass
    def stop(self):
      pass
  return TorchProfiler()


if nvtx_enabled:
  import nvtx
  nvtx_start = nvtx_start
  nvtx_stop = nvtx_stop
else:
  nvtx = no_op_nvtx
  nvtx_start = noop
  nvtx_stop = noop

if torchprofiler_enabled:
  TorchProfiler = torch_profiler
else:
  TorchProfiler = no_op_torch_profiler