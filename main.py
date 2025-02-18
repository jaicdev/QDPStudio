import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import copy

from model_loader import load_model
from compression.pruning import Pruning
from evaluation import Evaluation
from compression.quantization import Quantization
from compression.decomposition import Decomposition
from compression.knowledge_distillation import KnowledgeDistillation

class ModelCompressionFramework:
    def __init__(self, model=None, model_name=None, dataset_name=None, batch_size=128, custom_evaluation_fn=None, device=None, accuracy_tolerance=5.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(model=model, model_name=model_name, device=self.device)

        # Load dataset based on the dataset name argument
        if dataset_name:
            self.train_loader, self.val_loader = self.load_dataset(dataset_name, batch_size)
        else:
            raise ValueError("A dataset name must be provided.")

        self.custom_evaluation_fn = custom_evaluation_fn or Evaluation(self.device).default_evaluation
        self.accuracy_tolerance = accuracy_tolerance
        self.compressed_models = {}
        self.original_metrics = {}
        self.compressed_metrics = {}

    def load_dataset(self, dataset_name, batch_size):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_name.upper() == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name.upper() == 'MNIST':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name.upper() == 'IMAGENET':
            train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
            val_dataset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: CIFAR10, MNIST, ImageNet.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, num_epochs=150, learning_rate=0.0001):
        print("Training the model on the dataset...")

        # Set up training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        print("Finished Training")

    def compress_model(self, prune, quantize, decompose, all_compressions):
        # Save the original model metrics
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
            quantization.fixed_point_quantization()
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
        # Train the model first
        self.train_model(num_epochs=5)  # Train for 5 epochs

        # Running compression pipeline
        print("Running Compression Pipeline...")
        self.compress_model(prune, quantize, decompose, all_compressions)

        # Compare results of different compression methods
        self.compare_results()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run model compression with specified techniques.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for training and validation (e.g., CIFAR10, MNIST, ImageNet)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--prune', action='store_true', help='Perform pruning on the model')
    parser.add_argument('--quantize', action='store_true', help='Perform quantization on the model')
    parser.add_argument('--decompose', action='store_true', help='Perform decomposition on the model')
    parser.add_argument('--all', action='store_true', help='Perform all compression techniques and compare results')

    args = parser.parse_args()

    # Load model and run compression pipeline with the specified dataset and options
    model = load_model(model_name="resnet18")
    framework = ModelCompressionFramework(model=model, dataset_name=args.dataset, batch_size=args.batch_size)
    framework.run_pipeline(prune=args.prune, quantize=args.quantize, decompose=args.decompose, all_compressions=args.all)

