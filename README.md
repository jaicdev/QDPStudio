# QDP Studio

QDP Studio is a comprehensive model compression framework designed to optimize deep learning models through multiple advanced techniques: **Quantization**, **Decomposition**, and **Pruning**. With support for hybrid compression, QDP Studio enables you to significantly reduce model size, accelerate inference, and maintain high accuracy—all while streamlining deployment across various devices.

---

## Features

- **Quantization**  
  Leverage dynamic, static, and quantization-aware training (QAT) techniques to convert high-precision models into lower-bit representations for faster, more efficient inference. citeturn0file0

- **Pruning**  
  Reduce model complexity by removing redundant weights using both unstructured and structured pruning methods, combined with iterative fine-tuning to preserve performance. citeturn0file1

- **Decomposition**  
  Utilize techniques such as Truncated SVD to approximate and simplify model layers, decreasing computational load without sacrificing accuracy. citeturn0file3

- **Knowledge Distillation**  
  Optionally integrate teacher-student training methods to further compress and optimize models by transferring knowledge from a larger, pre-trained network. citeturn0file2

- **Hybrid Compression Pipeline**  
  Apply all supported compression techniques sequentially in one unified pipeline. This hybrid approach maximizes the benefits of each method, ensuring optimal trade-offs between efficiency and accuracy.

- **Comprehensive Evaluation**  
  Evaluate models using key metrics including accuracy, inference time, and model size, allowing for direct comparison between the original and compressed versions.

---

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch & Torchvision
- TIMM (for additional model support)
- Transformers (for Hugging Face models)
- scikit-learn
- tensorly
- Additional libraries: argparse, yaml, logging, wandb, etc.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/qdp-studio.git
   cd qdp-studio
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**

   Edit the `config.yaml` file to set model parameters, device preference, batch size, learning rate, and the number of epochs. citeturn1file1

---

## Usage

### Command-Line Interface

QDP Studio is controlled via `main.py`, which offers a command-line interface to select the dataset and compression techniques you wish to apply.

**Example Command:**

```bash
python main.py --dataset CIFAR10 --prune --quantize --decompose
```

This command will:
- Train a model (default: ResNet18) on the CIFAR10 dataset.
- Apply pruning, quantization, and decomposition.
- Evaluate and compare the performance of the original and each compressed variant.

**Available Arguments:**

- `--dataset`: Specify the dataset (e.g., CIFAR10, MNIST, ImageNet).
- `--batch_size`: Define the batch size for training and evaluation.
- `--prune`: Apply pruning.
- `--quantize`: Apply quantization.
- `--decompose`: Apply decomposition.
- `--all`: Run all compression techniques as a hybrid approach and compare results.

### Running Hybrid Compression

Hybrid compression in QDP Studio allows you to apply all the supported techniques in sequence to maximize model optimization.

#### How Hybrid Compression Works

1. **Model Training:**  
   The framework begins by training the base model on your chosen dataset, ensuring a strong initial performance.

2. **Sequential Compression:**
   - **Pruning:** Removes redundant weights to reduce overall complexity.
   - **Quantization:** Converts model weights to lower-precision formats, enhancing efficiency.
   - **Decomposition:** Simplifies model layers using techniques like truncated SVD to cut down on computational demands.

3. **Post-Compression Fine-Tuning:**  
   Each compression stage is followed by fine-tuning to mitigate any loss in accuracy, with customizable epochs and learning rates.

4. **Evaluation:**  
   After applying hybrid compression, the framework evaluates the model’s performance—comparing accuracy, inference time, and model size between the original and compressed versions.

#### Running the Hybrid Pipeline

To execute the full hybrid compression pipeline, use the `--all` flag:

```bash
python main.py --dataset CIFAR10 --all
```

This command will:
- Train the default model on the specified dataset.
- Sequentially apply pruning, quantization, and decomposition.
- Fine-tune after each step, and evaluate performance across all compression techniques.

#### Customization

- **Fine-Tuning Parameters:**  
  Adjust the number of epochs and learning rate for each fine-tuning stage directly in the source code or via configuration options.

- **Compression Settings:**  
  Modify pruning rates, quantization backends, and decomposition ranks in the `config.yaml` or within the respective modules to best suit your model and deployment requirements.

- **Evaluation Metrics:**  
  The framework logs key metrics after each compression step, providing a clear comparison to help optimize your compression strategy.

---

## Acknowledgements

- Supported from the Science, Technology, and Innovation (STI) Policy of Gujarat Council of Science and Technology, Department of Science and Technology, Government of the Gujarat State, India (Grant Number: GUJCOST/STI/2021-22/3858) project "Person Retrieval in Video Surveillance".
- Thanks to the PyTorch and Torchvision communities for their excellent tools and documentation.
- Inspired by various model compression techniques and research in the field of deep learning.

---

## Contributing

Contributions are welcome! To get started:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push the branch: `git push origin feature/my-new-feature`
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
