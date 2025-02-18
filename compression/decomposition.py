import torch
import numpy as np
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from tensorly.decomposition import parafac, tucker
import tensorly as tl

class Decomposition:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def truncated_svd_approximation(self, rank=10):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    weight = module.weight.data
                    weight_np = weight.cpu().numpy()
                    svd = TruncatedSVD(n_components=rank, random_state=42)
                    reduced_matrix = svd.fit_transform(weight_np)
                    weight_approx_np = svd.inverse_transform(reduced_matrix)
                    module.weight.data = torch.tensor(weight_approx_np, device=self.device, dtype=weight.dtype)

