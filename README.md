# QDP Studio

QDP Studio is a comprehensive model compression framework designed to optimize deep learning models through multiple advanced techniques: **Quantization**, **Decomposition**, **Pruning**, and **Knowledge Distillation**. With support for hybrid compression, QDP Studio enables you to significantly reduce model size, accelerate inference, and maintain high accuracy—all while streamlining deployment across various devices.

---

## Features

- **Quantization**  
  Leverage dynamic, static, and quantization-aware training (QAT) techniques to convert high-precision models into lower-bit representations for faster, more efficient inference.  

- **Pruning**  
  Reduce model complexity by removing redundant weights with both unstructured and structured pruning methods, combined with iterative fine-tuning to preserve performance.  

- **Decomposition**  
  Utilize techniques such as Truncated SVD to approximate and simplify model layers, decreasing computational load without sacrificing accuracy.  

- **Knowledge Distillation**  
  Optionally integrate teacher-student training methods to further compress and optimize models by transferring knowledge from a larger, pre-trained network.  

- **Hybrid Compression Pipeline**  
  Apply all supported compression techniques sequentially in one unified pipeline. This hybrid approach maximizes the benefits of each method, ensuring optimal trade-offs between efficiency and accuracy.

- **Comprehensive Evaluation**  
  Evaluate models using key metrics—including accuracy, inference time, and model size—to directly compare the original and compressed versions.

- **Custom Model & Dataset Support**  
  Import and utilize your own custom models and datasets. Provide a custom model file path or a custom Python dataset module (which must implement a `get_custom_dataset()` function returning `(train_dataset, val_dataset)`).

---

## Getting Started

### Prerequisites

- Python 3.7+
- [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html)
- [TIMM](https://github.com/rwightman/pytorch-image-models) for additional model support
- [Transformers](https://huggingface.co/transformers/) for Hugging Face models
- [scikit-learn](https://scikit-learn.org/)
- [tensorly](https://tensorly.org/stable/)
- Additional libraries: `argparse`, `pyyaml`, `logging`, `wandb`, etc.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/jaicdev/QDPStudio.git
   cd QDPStudio
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**

   Edit the `config.yaml` file to set model parameters, device preference, batch size, learning rate, number of epochs, and compression settings (e.g., prune ratio). Example:

   ```yaml
   device: "cuda"      # Options: "cuda", "cpu", "mps"
   model_name: "resnet18"
   pretrained: true
   hf_model_name: null
   timm_model_name: null
   prune_ratio: 0.2
   ```

---

## Usage

### Command-Line Interface

QDP Studio is controlled via `main.py`, which provides a command-line interface to select the dataset and compression techniques.

**Example Command:**

```bash
python main.py --dataset CIFAR10 --prune --quantize --decompose
```

This command will:
- Train a model (default: ResNet18) on the CIFAR10 dataset.
- Apply pruning, quantization, and decomposition.
- Evaluate and compare the performance of the original and compressed model variants.

**Key Arguments:**

- `--dataset`: Specify the standard dataset (e.g., CIFAR10, MNIST, ImageNet).
- `--custom_dataset`: Python module name for a custom dataset (must implement a `get_custom_dataset()` function).
- `--batch_size`: Define the batch size for training and evaluation.
- `--custom_model`: Path to a custom model file (overrides standard model loading via `--model_name`).
- `--model_name`: Name of a torchvision model (default: "resnet18").
- `--prune`: Apply pruning.
- `--quantize`: Apply quantization.
- `--decompose`: Apply decomposition.
- `--all`: Run all compression techniques sequentially (hybrid approach).
- `--num_epochs`: Number of training epochs.

### Using a Custom Model

If you have a custom model file, use the `--custom_model` argument:

```bash
python main.py --custom_model path/to/your/custom_model.pth --dataset CIFAR10 --prune --quantize --decompose --num_epochs 5
```

### Using a Custom Dataset

Create a Python module (e.g., `my_dataset.py`) that implements a `get_custom_dataset()` function. For example:

```python
def get_custom_dataset():
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor
    train_dataset = FakeData(transform=ToTensor())
    val_dataset = FakeData(transform=ToTensor())
    return train_dataset, val_dataset
```

Then run:

```bash
python main.py --custom_dataset my_dataset --custom_model path/to/your/custom_model.pth --prune --quantize --decompose --num_epochs 5
```

---

## Hybrid Compression Pipeline

Hybrid compression applies all supported techniques sequentially:

1. **Model Training:**  
   Train the base model on your chosen dataset to ensure strong initial performance.

2. **Sequential Compression:**
   - **Pruning:** Remove redundant weights to reduce complexity.
   - **Quantization:** Convert model weights to lower precision using dynamic, static, or QAT techniques.
   - **Decomposition:** Simplify model layers using methods like truncated SVD.

3. **Post-Compression Fine-Tuning:**  
   Fine-tune after each compression step to mitigate loss in accuracy.

4. **Evaluation:**  
   Compare key metrics—accuracy, inference time, and model size—between the original and compressed models.

**Run the Hybrid Pipeline using the `--all` flag:**

```bash
python main.py --dataset CIFAR10 --all
```

---

## Logging & Evaluation

- Logging is implemented via Python’s `logging` module, with optional integration using [Weights & Biases (wandb)](https://wandb.ai) for comprehensive experimental tracking.
- The framework evaluates models on metrics including accuracy, precision, recall, F1-score, and inference latency.
- Detailed logging enables monitoring the impact of each compression technique.

---

## Troubleshooting & Tips

- **Configuration Issues:**  
  Ensure your `config.yaml` is properly formatted. Invalid configurations will result in runtime errors.
  
- **Custom Module Integration:**  
  Verify that any custom dataset module is on your Python path and implements the required `get_custom_dataset()` function.
  
- **Fusion Configurations:**  
  For optimal quantization, consider defining a custom fusion configuration mapping if the default does not meet your model's needs.
  
- **Testing:**  
  It is recommended to perform end-to-end tests to ensure that all components integrate seamlessly within the compression pipeline.

---

## Acknowledgements

- Supported by the Science, Technology, and Innovation (STI) Policy of Gujarat Council of Science and Technology, Department of Science and Technology, Government of Gujarat, India (Grant Number: GUJCOST/STI/2021-22/3858).
- Special thanks to the communities behind [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html), [TIMM](https://github.com/rwightman/pytorch-image-models), and [Transformers](https://huggingface.co/transformers/).
- **If you use this repository in your work, please cite:**

```bibtex
@ARTICLE{chaudhari2025onboard,
  author={Chaudhari, Jay N. and Galiyawala, Hiren and Sharma, Paawan and Shukla, Pancham and Raval, Mehul S.},
  journal={IEEE Access}, 
  title={Onboard Person Retrieval System With Model Compression: A Case Study on Nvidia Jetson Orin AGX}, 
  year={2025},
  volume={13},
  number={},
  pages={8257-8269},
  doi={10.1109/ACCESS.2025.3527134},
  ISSN={2169-3536},
  month={}
}
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -am 'Add new feature'
   ```
4. Push the branch:
   ```bash
   git push origin feature/my-new-feature
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
