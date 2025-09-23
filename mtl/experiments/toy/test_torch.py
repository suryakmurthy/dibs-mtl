import torch
print(torch.__version__)
print(torch.version.cuda)   # CUDA version it was built against
print(torch.backends.cudnn.version())  # cuDNN version (if available)
