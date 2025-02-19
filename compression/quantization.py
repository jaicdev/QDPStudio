import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert, prepare_qat
from utils import train_model
import warnings

class Quantization:
    def __init__(self, model, device, fusion_config=None, backend="fbgemm"):
        """
        Initializes the Quantization class.

        Parameters:
        - model: The neural network model to quantize.
        - device: The device (CPU/GPU) to use.
        - fusion_config: A dictionary specifying groups of modules to fuse. Ensure the modules listed in each group are fusion-eligible.
        - backend: The quantized backend to use. Supported backends: "fbgemm", "qnnpack".
        """
        self.model = model.to(device)
        self.device = device
        self.fusion_config = fusion_config or {}
        self.set_quantization_backend(backend)
        self.supported_backends = ["fbgemm", "qnnpack"]

        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported quantization backend: {backend}. Supported backends: {self.supported_backends}")

    def _validate_fusion_config(self):
        """
        Validates the fusion configuration by checking if the modules in each group are eligible for fusion.
        Returns a new fusion configuration dictionary with only the valid groups.
        """
        valid_fusion_config = {}
        for group_name, modules in self.fusion_config.items():
            valid = True
            for module_name in modules:
                # Verify the module exists in the model
                try:
                    module = dict(self.model.named_modules())[module_name]
                    # For fusion, we typically require Conv/BatchNorm/ReLU groups.
                    if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                        valid = False
                        warnings.warn(f"Module {module_name} in group '{group_name}' is not eligible for fusion.")
                        break
                except KeyError:
                    valid = False
                    warnings.warn(f"Module {module_name} in group '{group_name}' not found in the model.")
                    break

            if valid:
                valid_fusion_config[group_name] = modules

        return valid_fusion_config

    def apply_dynamic_quantization(self):
        """
        Apply dynamic quantization to nn.Linear layers.
        """
        self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        return self.model

    def apply_static_quantization(self, calibration_data, error_threshold=0.2):
        """
        Apply static quantization to the model. This requires calibration data.
        
        Process:
        - Validate and perform module fusion according to a validated fusion configuration.
        - Set the qconfig using default_qconfig.
        - Prepare the model for calibration.
        - Iterates over calibration_data and counts errors.
          If calibration errors exceed error_threshold * num_batches, raises an exception.
        - Converts the model after calibration.

        Parameters:
        - calibration_data: An iterable of calibration batches.
        - error_threshold: Maximum fraction of calibration batches that may fail before aborting.
        """
        # Validate and apply fusion
        valid_fusion_config = self._validate_fusion_config()
        if valid_fusion_config:
            for group_name, modules in valid_fusion_config.items():
                try:
                    self.model.fuse_modules(modules, inplace=True)
                except Exception as e:
                    warnings.warn(f"Fusion failed for group '{group_name}': {e}")
        else:
            warnings.warn("No valid fusion configuration provided. Skipping fusion.")

        self.model.qconfig = torch.quantization.default_qconfig
        try:
            prepared_model = prepare(self.model)
        except Exception as e:
            raise RuntimeError(f"Model preparation for static quantization failed: {e}")
        
        error_count = 0
        total_batches = 0
        with torch.no_grad():
            for data in calibration_data:
                total_batches += 1
                try:
                    prepared_model(data)
                except Exception as e:
                    error_count += 1
                    warnings.warn(f"Calibration step failed on a batch: {e}")
                    if (error_count / total_batches) > error_threshold:
                        raise RuntimeError("Too many calibration errors encountered. Aborting static quantization.")
        
        try:
            self.model = convert(prepared_model)
        except Exception as e:
            raise RuntimeError(f"Model conversion after static quantization failed: {e}")
        return self.model

    def apply_qat(self, train_loader, criterion, optimizer, num_epochs=5):
        """
        Apply Quantization-Aware Training (QAT) to the model.

        Process:
        - Validate and attempt to fuse modules using the fusion configuration if provided.
        - Set the model's qconfig to the default QAT configuration.
        - Prepare the model for QAT.
        - Train the model using provided training parameters.
        - Convert the model after QAT training.
        """
        valid_fusion_config = self._validate_fusion_config()
        if valid_fusion_config:
            for group_name, modules in valid_fusion_config.items():
                try:
                    self.model.fuse_modules(modules, inplace=True)
                except Exception as e:
                    warnings.warn(f"Fusion failed for group '{group_name}' during QAT: {e}")
        else:
            warnings.warn("No valid fusion configuration provided. Skipping fusion during QAT.")
                
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(self.device.type)
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

    def apply_mixed_precision(self, train_loader, criterion, optimizer, num_epochs=5):
        """
        Apply mixed precision training to the model during training.
        Mixed precision training leverages torch.cuda.amp to improve performance.
        """
        self.model = train_model(
            self.model, train_loader, criterion, optimizer, self.device, num_epochs, mixed_precision=True
        )
        return self.model

    def set_quantization_backend(self, backend="fbgemm"):
        """
        Set the quantization backend for the process.
        """
        torch.backends.quantized.engine = backend
