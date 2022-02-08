
tensorboard_output_dir = './log/torch_profiler'
nvtx_enabled = False
torchprofiler_enabled = True
print(f'nvtx_enabled: {nvtx_enabled}')
print(f'torchprofiler_enabled: {torchprofiler_enabled}')
print(f'tensorboard_output_dir: {tensorboard_output_dir}')

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
  # def trace_handler(prof):
  #   print(prof.key_averages().table(
  #       sort_by="self_cuda_time_total", row_limit=-1))
  import torch
  prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_output_dir),
        # on_trace_ready=trace_handler, # cool but not needed now
        # profile_memory=True, # makes it slower
        # 
        record_shapes=True,
        with_stack=True
  )
  return prof

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