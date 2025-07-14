import torch
print("torch version:", torch.__version__)
print("hasattr(torch.backends, 'mps'):", hasattr(torch.backends, 'mps'))
if hasattr(torch.backends, 'mps'):
    print("torch.backends.mps.is_available():", torch.backends.mps.is_available())
    print("torch.backends.mps.is_built():", torch.backends.mps.is_built())
else:
    print("MPS backend not present in torch.backends")