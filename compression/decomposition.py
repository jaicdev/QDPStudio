import torch
import torch.nn as nn
import logging

class Decomposition:
    def __init__(self, model, device):
        """
        Initializes the Decomposition class.

        Parameters:
        - model: The neural network model to be decomposed.
        - device: The device to perform operations on.
        """
        self.model = model.to(device)
        self.device = device

    def truncated_svd_approximation(self, rank=10):
        """
        Apply truncated SVD on eligible Linear layers using torch.linalg.svd.
        This method approximates the weight matrix by keeping only the top 'rank' singular values.

        Parameters:
        - rank: Desired rank for the low-rank approximation.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                try:
                    # Use torch.linalg.svd for decomposition
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    effective_rank = min(rank, U.shape[1])
                    U_approx = U[:, :effective_rank]
                    S_approx = S[:effective_rank]
                    Vh_approx = Vh[:effective_rank, :]
                    approx_weight = U_approx @ torch.diag(S_approx) @ Vh_approx
                    module.weight.data.copy_(approx_weight)
                    logging.info(f"Linear layer '{name}' decomposed to rank {effective_rank} using truncated SVD.")
                except Exception as e:
                    logging.warning(f"Truncated SVD failed for layer '{name}': {e}")
        return self.model

    def conv_tucker_decomposition(self, rank=None):
        """
        Apply Tucker Decomposition on eligible Conv2d layers.
        This method performs a low-rank approximation on the convolutional filters,
        decomposing them into a core tensor and factor matrices.

        Note:
        - This implementation uses the TensorLy library.
          Ensure TensorLy is installed ('pip install tensorly').

        Parameters:
        - rank: Tuple specifying the target rank for each mode. If not provided,
                default ranks will be used.
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import tucker
            tl.set_backend('pytorch')
        except ImportError:
            logging.warning("TensorLy is not installed. Tucker decomposition cannot be performed.")
            return self.model

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                # Determine default rank if not provided
                if rank is None:
                    # For example, use full in_channels and kernel size, and reduce output channels by half
                    rank = (weight.shape[1], weight.shape[2], weight.shape[0] // 2)
                try:
                    core, factors = tucker(weight, ranks=rank)
                    # Reconstruct the approximated weight from Tucker decomposition
                    approx_weight = tl.tucker_to_tensor((core, factors))
                    module.weight.data.copy_(approx_weight)
                    logging.info(f"Conv2d layer '{name}' decomposed using Tucker decomposition with rank {rank}.")
                except Exception as e:
                    logging.warning(f"Tucker decomposition failed for layer '{name}': {e}")
        return self.model

    def conv_cp_decomposition(self, rank=None):
        """
        Apply CP (CANDECOMP/PARAFAC) Decomposition on eligible Conv2d layers.
        This method decomposes the convolutional filters into a sum of rank-one tensors.

        Note:
        - This implementation uses the TensorLy library.
          Ensure TensorLy is installed ('pip install tensorly').

        Parameters:
        - rank: An integer specifying the desired CP rank. If not provided,
                a default value based on the weight dimensions will be used.
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac
            tl.set_backend('pytorch')
        except ImportError:
            logging.warning("TensorLy is not installed. CP decomposition cannot be performed.")
            return self.model

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                if rank is None:
                    rank = min(weight.shape) // 2  # default rank as half of the smaller dimension
                try:
                    factors = parafac(weight, rank=rank, init='svd')
                    approx_weight = tl.kruskal_to_tensor(factors)
                    module.weight.data.copy_(approx_weight)
                    logging.info(f"Conv2d layer '{name}' decomposed using CP decomposition with rank {rank}.")
                except Exception as e:
                    logging.warning(f"CP decomposition failed for layer '{name}': {e}")
        return self.model

    def decompose(self, method='svd', **kwargs):
        """
        General method for model decomposition.

        Parameters:
        - method: The decomposition method to use. Options:
                  'svd'    - Truncated SVD for Linear layers.
                  'tucker' - Tucker decomposition for Conv2d layers.
                  'cp'     - CP decomposition for Conv2d layers.
        - kwargs: Additional parameters passed to the specific decomposition method.

        Returns:
        - The model with decomposed weights.
        """
        if method == 'svd':
            return self.truncated_svd_approximation(**kwargs)
        elif method == 'tucker':
            return self.conv_tucker_decomposition(**kwargs)
        elif method == 'cp':
            return self.conv_cp_decomposition(**kwargs)
        else:
            logging.warning(f"Unknown decomposition method '{method}'. No decomposition applied.")
            return self.model
