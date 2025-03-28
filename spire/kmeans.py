import joblib
import numpy as np
import torch


class KmeansForInference:
    """
    Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_km_label.py,
    copyright (c) Facebook, Inc. and its affiliates.

    K-means adapted from fairseq. Note that using the torch backend may
    (very rarely) produce slightly different cluster assignments than using
    numpy due to numerical differences.
    """
    def __init__(self, km_path, device="cpu", backend="torch", torch_dtype=torch.float32):
        self.backend = backend

        km_model = joblib.load(km_path)
        self.C = km_model.cluster_centers_.transpose()
        self.Cnorm = (self.C ** 2).sum(0, keepdims=True)
        if self.backend == "torch":
            device = torch.device(device)
            self.C = torch.from_numpy(self.C).to(device=device, dtype=torch_dtype)
            self.Cnorm = torch.from_numpy(self.Cnorm).to(device=device, dtype=torch_dtype)

    def __call__(self, x):
        if self.backend == "torch":
            dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            return dist.argmin(dim=1)
        else:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C)
                + self.Cnorm
            )
            return np.argmin(dist, axis=1)
