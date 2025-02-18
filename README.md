# QDP Studio

QDP Studio is a unified framework for deep learning model compression. It combines quantization, pruning, decomposition, and knowledge distillation to reduce model size, improve inference speed, and maintain accuracy. This streamlined pipeline for training, compressing, and evaluating models is designed to optimize deployments in resource-constrained environments.

## Features

- **Dynamic Quantization:** Automatically apply dynamic quantization to reduce model size, focusing on environments where runtime speed is critical.
- **Static Quantization & QAT:** Fuse model modules based on a configurable fusion strategy for optimal static quantization and quantization-aware training (QAT).
- **Pruning:** Apply unstructured weight pruning on convolutional and linear layers, with configurable prune ratios and logging of model sparsity.
- **Decomposition:** Use truncated SVD to approximate weight matrices in eligible layers (e.g., `nn.Linear`), with configurable rank options.
- **Knowledge Distillation:** Distill knowledge from a larger teacher model to a smaller student model using a blended loss (cross-entropy coupled with KL divergence). Hyperparameters for temperature and blending ratio are configurable.
- **Custom Model & Dataset Support:** Easily integrate custom models and datasets by providing a model file path or custom Python module.

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- torchvision
- timm
- transformers
- Other dependencies: PyYAML, wandb, scikit-learn

You can install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

### Configuration

The framework uses a `config.yaml` file for configuration. Ensure that this file exists in the root directory. Example settings include:

```yaml
device: "cuda"      # or "cpu", "mps" depending on your hardware
model_name: "resnet18"
pretrained: true
hf_model_name: null
timm_model_name: null
prune_ratio: 0.2
```

### Running the Pipeline

To run the compression pipeline using standard datasets (such as CIFAR10, MNIST, or ImageNet), execute:

```bash
python main.py --dataset CIFAR10 --model_name resnet18 --prune --quantize --decompose --num_epochs 5
```

### Using a Custom Model

If you have a custom model file, supply its path using the `--custom_model` argument:

```bash
python main.py --custom_model path/to/your/custom_model.pth --dataset CIFAR10 --prune --quantize --decompose --num_epochs 5
```

### Using a Custom Dataset

If you want to use a custom dataset, create a Python module that implements a function named `get_custom_dataset()` which returns a tuple `(train_dataset, val_dataset)`. For example, your custom module (e.g., `my_dataset.py`) could look like:

```python
def get_custom_dataset():
    # Your custom dataset implementation here
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor
    train_dataset = FakeData(transform=ToTensor())
    val_dataset = FakeData(transform=ToTensor())
    return train_dataset, val_dataset
```

Then run the pipeline with:

```bash
python main.py --custom_dataset my_dataset --custom_model path/to/your/custom_model.pth --prune --quantize --decompose --num_epochs 5
```

### Understanding the Compression Modules

- **Quantization:** Review `compression/quantization.py` for configurations like fusion strategies and calibration steps.
- **Pruning:** Check `compression/pruning.py` to configure your prune ratio and to log model sparsity.
- **Decomposition:** Open `compression/decomposition.py` for details on the SVD approximation and setting the rank.
- **Knowledge Distillation:** Take a look at `compression/knowledge_distillation.py` to adjust distillation parameters such as temperature and alpha.
```` 

## Logging & Evaluation

Logging is implemented using Python's logging module and, optionally, [Weights & Biases (wandb)](https://wandb.ai). Ensure wandb is correctly configured if you choose to leverage its logging capabilities. Evaluation metrics (accuracy, precision, recall, F1-score, latency) are computed automatically post-training and post-compression.

## Contributing

Contributions are welcome! Please follow the repository guidelines and ensure that any changes are thoroughly tested with the entire pipeline.

## License

[MIT License](LICENSE)

## Acknowledgments

QDP Studio leverages state-of-the-art libraries including PyTorch, torchvision, timm, and transformers. Special thanks to the contributors and the open-source community.
