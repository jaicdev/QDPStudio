import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import copy
import importlib
import os

from model_loader import load_model
from compression.pruning import Pruning
from evaluation import Evaluation
from compression.quantization import Quantization
from compression.decomposition import Decomposition
from compression.knowledge_distillation import KnowledgeDistillation

class ModelCompressionFramework:
    def __init__(self, model=None, model_name=None, custom_model_path=None,
                 dataset_name=None, custom_dataset_module=None,
                 batch_size=128, custom_evaluation_fn=None, device=None,
                 accuracy_tolerance=5.0, num_epochs=150, log_frequency=100,
                 checkpoint_path=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Load custom model if provided, otherwise use model_name
        self.model = load_model(model=model, model_name=model_name, custom_model_path=custom_model_path, device=self.device)
        self.custom_dataset_module = custom_dataset_module
        
        # Load dataset: custom dataset or standard dataset
        if custom_dataset_module:
            # Expect the custom dataset module to implement get_custom_dataset() returning (train_dataset, val_dataset)
            custom_mod = importlib.import_module(custom_dataset_module)
            if hasattr(custom_mod, "get_custom_dataset"):
                train_dataset, val_dataset = custom_mod.get_custom_dataset()
            else:
                raise ValueError("The custom dataset module must implement a 'get_custom_dataset' function.")
        elif dataset_name:
            train_dataset, val_dataset = self.load_standard_dataset(dataset_name, batch_size)
        else:
            raise ValueError("Either a dataset name or a custom dataset module must be provided.")
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.custom_evaluation_fn = custom_evaluation_fn or Evaluation(self.device).default_evaluation
        self.accuracy_tolerance = accuracy_tolerance
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.checkpoint_path = checkpoint_path
        self.compressed_models = {}
        self.original_metrics = {}
        self.compressed_metrics = {}

    def load_standard_dataset(self, dataset_name, batch_size):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_type = dataset_name.upper()
        if dataset_type == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_type == 'MNIST':
            transform_mnist = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
            val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        elif dataset_type == 'IMAGENET':
            train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
            val_dataset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: CIFAR10, MNIST, IMAGENET.")
        return train_dataset, val_dataset

    def train_model(self, num_epochs=None, learning_rate=0.0001):
        num_epochs = num_epochs or self.num_epochs
        print("Training the model on the dataset...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % self.log_frequency == 0:
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / self.log_frequency:.3f}")
                    running_loss = 0.0

            # Optional: Save checkpoint after each epoch if checkpoint_path is provided.
            if self.checkpoint_path:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                checkpoint_file = os.path.join(self.checkpoint_path, f"model_epoch_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_file)
                print(f"Checkpoint saved: {checkpoint_file}")
        print("Finished Training")

    def compress_model(self, prune, quantize, decompose, all_compressions):
        self.original_metrics = self.custom_evaluation_fn(self.model, self.val_loader)
        if prune or all_compressions:
            print("Applying Pruning...")
            pruned_model = copy.deepcopy(self.model)
            pruning = Pruning(pruned_model, self.device)
            pruning.weight_pruning()
            self.compressed_models['pruned'] = pruned_model
            self.compressed_metrics['pruned'] = self.custom_evaluation_fn(pruned_model, self.val_loader)
        if quantize or all_compressions:
            print("Applying Quantization...")
            quantized_model = copy.deepcopy(self.model)
            quantization = Quantization(quantized_model, self.device)
            try:
                quantization.apply_dynamic_quantization()
            except Exception as e:
                print(f"Quantization failed: {e}")
            self.compressed_models['quantized'] = quantized_model
            self.compressed_metrics['quantized'] = self.custom_evaluation_fn(quantized_model, self.val_loader)
        if decompose or all_compressions:
            print("Applying Decomposition...")
            decomposed_model = copy.deepcopy(self.model)
            decomposition = Decomposition(decomposed_model, self.device)
            decomposition.truncated_svd_approximation(rank=10)
            self.compressed_models['decomposed'] = decomposed_model
            self.compressed_metrics['decomposed'] = self.custom_evaluation_fn(decomposed_model, self.val_loader)

    def compare_results(self):
        print("\nComparison of Compression Methods:")
        print(f"Original Metrics: {self.original_metrics}")
        for method, metrics in self.compressed_metrics.items():
            print(f"{method.capitalize()} Metrics: {metrics}")

    def run_pipeline(self, prune, quantize, decompose, all_compressions):
        self.train_model()
        print("Running Compression Pipeline...")
        self.compress_model(prune, quantize, decompose, all_compressions)
        self.compare_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model compression with specified techniques.")
    parser.add_argument('--dataset', type=str, help='Dataset to use for training and validation (e.g., CIFAR10, MNIST, IMAGENET)')
    parser.add_argument('--custom_dataset', type=str, help='Python module path for custom dataset (must implement get_custom_dataset())')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--custom_model', type=str, help='Path to a custom model file to load')
    parser.add_argument('--model_name', type=str, default="resnet18", help='Model name for torchvision models')
    parser.add_argument('--prune', action='store_true', help='Perform pruning on the model')
    parser.add_argument('--quantize', action='store_true', help='Perform quantization on the model')
    parser.add_argument('--decompose', action='store_true', help='Perform decomposition on the model')
    parser.add_argument('--all', action='store_true', help='Perform all compression techniques')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs (adjust as needed)')
    parser.add_argument('--log_frequency', type=int, default=100, help='Logging frequency (in number of batches)')
    parser.add_argument('--checkpoint_path', type=str, help='Directory path to save model checkpoints')
    
    args = parser.parse_args()
    
    framework = ModelCompressionFramework(
        model_name=args.model_name,
        custom_model_path=args.custom_model,
        dataset_name=args.dataset,
        custom_dataset_module=args.custom_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_frequency=args.log_frequency,
        checkpoint_path=args.checkpoint_path
    )
    framework.run_pipeline(prune=args.prune, quantize=args.quantize, decompose=args.decompose, all_compressions=args.all)
