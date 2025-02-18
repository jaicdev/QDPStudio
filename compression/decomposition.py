import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Decomposition:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def truncated_svd_approximation(self, rank=10):
        """
        Applies truncated SVD approximation on eligible layers.
        The method iterates over layers and approximates weight matrices.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                try:
                    U, S, V = torch.svd(weight)
                    # Determine effective rank based on provided rank or energy threshold
                    effective_rank = min(rank, U.shape[1])
                    U_approx = U[:, :effective_rank]
                    S_approx = S[:effective_rank]
                    V_approx = V[:, :effective_rank]
                    # Reconstruct approximated weight matrix: U * diag(S) * V^T
                    approx_weight = torch.mm(U_approx, torch.mm(torch.diag(S_approx), V_approx.t()))
                    module.weight.data.copy_(approx_weight)
                    logging.info(f"Layer {name} decomposed to rank {effective_rank}.")
                except Exception as e:
                    logging.warning(f"Decomposition failed for layer {name}: {e}")
            # Extend to other layer types if needed (e.g., Conv2d)
        return self.model
