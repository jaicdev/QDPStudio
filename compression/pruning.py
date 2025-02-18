import torch
import torch.nn as nn
import copy
import logging

class Pruning:
    def __init__(self, model, device, prune_ratio=0.2):
        self.model = model.to(device)
        self.device = device
        # prune_ratio: fraction of weights expected to be pruned
        self.prune_ratio = prune_ratio

    def weight_pruning(self):
        """
        Perform unstructured weight pruning on linear and convolutional layers.
        Logs the sparsity after pruning.
        """
        total_params = 0
        zero_params = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                total_params += weight.numel()
                # Create a mask for small values
                threshold = weight.abs().mean() * self.prune_ratio
                mask = weight.abs() > threshold
                module.weight.data.mul_(mask.float())
                zero_params += (~mask).sum().item()
        sparsity = zero_params / total_params if total_params > 0 else 0
        logging.info(f"Pruning applied. Model sparsity: {sparsity:.2%}")
        return self.model
