import torch

def shuffle_tensor(x, dim=0):
    indices = torch.randperm(x.shape[dim])
    new_x = torch.index_select(x, dim, indices)
    return new_x