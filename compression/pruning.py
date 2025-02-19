import torch
import torch.nn as nn
import logging

class Pruning:
    def __init__(self, model, device, prune_ratio=0.2, prune_bias=False):
        """
        Initializes the Pruning class with a given model.

        Parameters:
        - model: The neural network model to be pruned.
        - device: The device (CPU/GPU) to use.
        - prune_ratio: The target fraction of weights/filters/neuron outputs to prune.
        - prune_bias: Whether to also prune bias parameters.
        """
        self.model = model.to(device)
        self.device = device
        self.prune_ratio = prune_ratio
        self.prune_bias = prune_bias

    def weight_pruning(self):
        """
        Unstructured pruning for Conv2d and Linear layers based on a layer-wise quantile threshold.
        This method prunes individual weights.
        """
        total_params = 0
        zero_params = 0

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Prune weights
                weight = module.weight.data
                total_params += weight.numel()
                threshold = torch.quantile(weight.abs(), self.prune_ratio)
                weight_mask = weight.abs() > threshold
                module.weight.data.mul_(weight_mask.float())
                zero_params += (~weight_mask).sum().item()

                # Optionally prune biases
                if self.prune_bias and module.bias is not None:
                    bias = module.bias.data
                    total_params += bias.numel()
                    bias_threshold = torch.quantile(bias.abs(), self.prune_ratio)
                    bias_mask = bias.abs() > bias_threshold
                    module.bias.data.mul_(bias_mask.float())
                    zero_params += (~bias_mask).sum().item()

        sparsity = (zero_params / total_params) if total_params > 0 else 0
        logging.info(f"Unstructured pruning applied. Overall model sparsity: {sparsity:.2%}")
        return self.model

    def filter_pruning(self):
        """
        Structured pruning for Conv2d layers by pruning entire filters based on their L1-norm.
        For each Conv2d layer, if a filter's L1 norm is below the layer-specific threshold,
        the entire filter is set to zero.
        """
        total_filters = 0
        pruned_filters = 0

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                # weight shape: (out_channels, in_channels, kH, kW)
                out_channels = weight.size(0)
                total_filters += out_channels

                # Compute L1 norm for each filter (over in_channels and kernel dimensions)
                l1_norm = weight.abs().view(out_channels, -1).sum(dim=1)
                threshold = torch.quantile(l1_norm, self.prune_ratio)
                # Create a mask for filters to keep
                filter_mask = l1_norm > threshold
                pruned_filters += (filter_mask == 0).sum().item()

                # Zero out the entire filter if it is below threshold
                for i in range(out_channels):
                    if not filter_mask[i]:
                        module.weight.data[i].zero_()
                        if self.prune_bias and module.bias is not None:
                            module.bias.data[i] = 0

        pruning_ratio = (pruned_filters / total_filters) if total_filters > 0 else 0
        logging.info(f"Filter pruning applied. Pruned filters: {pruned_filters}/{total_filters} ({pruning_ratio:.2%})")
        return self.model

    def neuron_pruning(self):
        """
        Structured pruning for Linear layers by pruning entire neurons based on their L1-norm.
        For each Linear layer, if a neuron's outgoing weight vector (a row in weight matrix)
        has an L1 norm below the layer-specific threshold, it is set to zero.
        """
        total_neurons = 0
        pruned_neurons = 0

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                # weight shape: (out_features, in_features)
                out_features = weight.size(0)
                total_neurons += out_features

                # Compute L1 norm for each neuron (row in the weight matrix)
                l1_norm = weight.abs().sum(dim=1)
                threshold = torch.quantile(l1_norm, self.prune_ratio)
                neuron_mask = l1_norm > threshold
                pruned_neurons += (neuron_mask == 0).sum().item()

                # Zero out neurons that didn't pass the threshold
                for i in range(out_features):
                    if not neuron_mask[i]:
                        module.weight.data[i].zero_()
                        if self.prune_bias and module.bias is not None:
                            module.bias.data[i] = 0

        pruning_ratio = (pruned_neurons / total_neurons) if total_neurons > 0 else 0
        logging.info(f"Neuron pruning applied on Linear layers. Pruned neurons: {pruned_neurons}/{total_neurons} ({pruning_ratio:.2%})")
        return self.model

    def global_pruning(self):
        """
        Global unstructured pruning applied across all eligible layers (Conv2d and Linear).
        This method determines a global threshold based on the aggregated weight values
        from all layers, and prunes weights that fall below this threshold.
        """
        # Gather all weights from eligible layers into one tensor
        all_weights = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                all_weights.append(module.weight.data.abs().flatten())
        if len(all_weights) == 0:
            logging.info("No eligible layers found for global pruning.")
            return self.model

        all_weights = torch.cat(all_weights)
        global_threshold = torch.quantile(all_weights, self.prune_ratio)

        total_params = 0
        zero_params = 0
        # Apply the global threshold to prune weights
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                total_params += weight.numel()
                mask = weight.abs() > global_threshold
                module.weight.data.mul_(mask.float())
                zero_params += (~mask).sum().item()

                if self.prune_bias and module.bias is not None:
                    bias = module.bias.data
                    total_params += bias.numel()
                    bias_mask = bias.abs() > global_threshold
                    module.bias.data.mul_(bias_mask.float())
                    zero_params += (~bias_mask).sum().item()

        sparsity = (zero_params / total_params) if total_params > 0 else 0
        logging.info(f"Global pruning applied. Overall model sparsity: {sparsity:.2%}")
        return self.model
