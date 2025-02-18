import torch
import torch.nn as nn
import logging

class Pruning:
    def __init__(self, model, device, prune_ratio=0.2):
        self.model = model.to(device)
        self.device = device
        # fraction of weights to prune
        self.prune_ratio = prune_ratio

    def weight_pruning(self):
        """
        Perform unstructured weight pruning on Conv2d and Linear layers.
        Logs the resulting sparsity.
        """
        total_params = 0
        zero_params = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                total_params += weight.numel()
                threshold = weight.abs().mean() * self.prune_ratio
                mask = weight.abs() > threshold
                module.weight.data.mul_(mask.float())
                zero_params += (~mask).sum().item()
        sparsity = zero_params / total_params if total_params > 0 else 0
        logging.info(f"Pruning applied. Model sparsity: {sparsity:.2%}")
        return self.model
