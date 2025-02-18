import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic, prepare, convert, prepare_qat
from utils import train_model
import warnings

class Quantization:
    def __init__(self, model, device, fusion_config=None):
        self.model = model.to(device)
        self.device = device
        # fusion_config: a dict mapping layer names/groups that should be fused
        # e.g., {"conv_bn": ["conv1", "bn1"], "conv_bn_relu": ["conv2", "bn2", "relu2"]}
        self.fusion_config = fusion_config or {}

    def apply_dynamic_quantization(self):
        """
        Apply dynamic quantization to nn.Linear layers.
        """
        self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        return self.model

    def apply_static_quantization(self, calibration_data):
        """
        Apply static quantization to the model. Requires calibration data.
        """
        # If fusion config provided, fuse modules accordingly
        if self.fusion_config:
            for group in self.fusion_config.values():
                try:
                    self.model.fuse_modules(group, inplace=True)
                except Exception as e:
                    warnings.warn(f"Fusion of {group} failed: {e}")
        else:
            try:
                # Fallback: attempt a default fusion
                self.model.fuse_modules(inplace=True)
            except Exception as e:
                warnings.warn(f"Default fusion failed: {e}")

        self.model.qconfig = torch.quantization.default_qconfig
        try:
            prepared_model = prepare(self.model)
        except Exception as e:
            raise RuntimeError(f"Model preparation for static quantization failed: {e}")
            
        # Calibration step
        with torch.no_grad():
            for data in calibration_data:
                try:
                    prepared_model(data)
                except Exception as e:
                    warnings.warn(f"Calibration step failed on a batch: {e}")
        try:
            self.model = convert(prepared_model)
        except Exception as e:
            raise RuntimeError(f"Model conversion after static quantization failed: {e}")
        return self.model

    def apply_qat(self, train_loader, criterion, optimizer, num_epochs=5):
        """
        Apply Quantization-Aware Training (QAT) to the model.
        """
        if self.fusion_config:
            for group in self.fusion_config.values():
                try:
                    self.model.fuse_modules(group, inplace=True)
                except Exception as e:
                    warnings.warn(f"Fusion of {group} failed: {e}")
        else:
            try:
                self.model.fuse_modules(inplace=True)
            except Exception as e:
                warnings.warn(f"Default fusion failed: {e}")
                
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        try:
            self.model = prepare_qat(self.model)
        except Exception as e:
            raise RuntimeError(f"Preparing QAT failed: {e}")
            
        self.model = train_model(self.model, train_loader, criterion, optimizer, self.device, num_epochs)
        try:
            self.model = convert(self.model)
        except Exception as e:
            raise RuntimeError(f"Converting model after QAT failed: {e}")
        return self.model

    def set_quantization_backend(self, backend="fbgemm"):
        torch.backends.quantized.engine = backend

    def apply_mixed_precision_training(self, train_loader, criterion, optimizer, num_epochs=5):
        self.model = train_model(self.model, train_loader, criterion, optimizer, self.device, num_epochs, mixed_precision=True)
        return self.model
