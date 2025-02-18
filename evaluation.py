import torch
import time

class Evaluation:
    def __init__(self, device):
        self.device = device

    def default_evaluation(self, model, data_loader):
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

        return {
            'accuracy': 100 * correct / total,
            'inference_time': total_time / len(data_loader),
            'model_size': sum(p.numel() for p in model.parameters()) * 4 / 1e6  # in MB
        }

