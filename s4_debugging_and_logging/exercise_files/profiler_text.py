import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

model = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     print("outputs:", model(inputs))

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))




prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34"))
prof.start()
print("outputs:", model(inputs))
prof.stop()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# prof.export_chrome_trace("trace2.json")