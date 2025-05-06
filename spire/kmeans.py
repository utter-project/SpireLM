import joblib
import numpy as np
import torch


# Seems like it might make sense to turn this into an nn module (easier compilation?)
class KmeansForInference:
    """
    Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_km_label.py,
    copyright (c) Facebook, Inc. and its affiliates.

    K-means adapted from fairseq. Note that using the torch backend may
    (very rarely) produce slightly different cluster assignments than using
    numpy due to numerical differences.
    """
    def __init__(self, km_path, device="cpu", torch_dtype=torch.float32):
        km_model = joblib.load(km_path)
        C = km_model.cluster_centers_.transpose()
        Cnorm = (C ** 2).sum(0, keepdims=True)

        device = torch.device(device)
        self.C = torch.from_numpy(C).to(device=device, dtype=torch_dtype)
        self.Cnorm = torch.from_numpy(Cnorm).to(device=device, dtype=torch_dtype)

    def __call__(self, x):
        """
        x: historically len x dim
        - Change the dimension for sum/argmin to -1
        - Then it should work for b x len x dim, I hope
        """
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=-1)
