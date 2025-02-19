import torch
import time

class Evaluation:
    def __init__(self, device):
        self.device = device

    def default_evaluation(self, model, data_loader):
        """
        Evaluates the model's accuracy, average inference time per batch,
        and estimates the model size dynamically by summing the memory 
        usage of each parameter (in MB). This accounts for different 
        parameter data types (e.g. quantized or mixed precision).
        """
        model.eval()
        correct, total, total_time = 0, 0, 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                start = time.perf_counter()
                outputs = model(images)
                total_time += (time.perf_counter() - start)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Calculate model size by summing the bytes of all parameters,
        # then converting to megabytes.
        model_size_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
        model_size_mb = model_size_bytes / 1e6

        return {
            'accuracy': 100 * correct / total,
            'inference_time': total_time / len(data_loader),
            'model_size': model_size_mb  # size in MB
        }
