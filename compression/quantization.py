import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic, prepare, convert, prepare_qat
from utils import train_model

class Quantization:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def apply_dynamic_quantization(self):
        """
        Apply dynamic quantization to the model.
        """
        self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        return self.model
    
    def apply_static_quantization(self, calibration_data):
        """
        Apply static quantization to the model. Requires calibration data.
        """
        self.model.fuse_modules()  # Fuse layers if needed
        self.model.qconfig = torch.quantization.default_qconfig
        prepared_model = prepare(self.model)
        
        # Calibration step
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        self.model = convert(prepared_model)
        return self.model
    
    def apply_qat(self, train_loader, criterion, optimizer, num_epochs=5):
        """
        Apply Quantization-Aware Training (QAT) to the model.
        """
        self.model.fuse_modules()  # Fuse layers if needed
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.model = prepare_qat(self.model)
        
        self.model = train_model(self.model, train_loader, criterion, optimizer, self.device, num_epochs)
        
        self.model = convert(self.model)
        return self.model
    
    def set_quantization_backend(self, backend="fbgemm"):
        """
        Set the quantization backend.
        """
        torch.backends.quantized.engine = backend
    
    def apply_mixed_precision_training(self, train_loader, criterion, optimizer, num_epochs=5):
        """
        Apply Mixed Precision Training (FP16/BF16) using Automatic Mixed Precision (AMP).
        """
        self.model = train_model(self.model, train_loader, criterion, optimizer, self.device, num_epochs, mixed_precision=True)
        return self.model
