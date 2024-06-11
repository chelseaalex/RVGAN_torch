import torch

filename = 'D:/Desktop/codebase/eyes/RVGAN-master/rvgan_torch/DRIVE.npz'
data = torch.load(filename)
print(data.keys())
