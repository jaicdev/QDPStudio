# QDP Studio

QDP Studio is a unified framework for deep learning model compression. It combines quantization, decomposition, and pruning—along with optional knowledge distillation—to reduce model size, boost inference speed, and maintain accuracy. Designed with a hybrid compression pipeline, QDP Studio is ideal for optimizing deployments in resource-constrained environments.

---

## Features

- **Quantization**  
  Apply dynamic, static, and quantization-aware training (QAT) techniques to convert high-precision models into efficient lower-precision representations.

- **Pruning**  
  Remove redundant weights using unstructured and structured pruning methods, with iterative fine-tuning to recover accuracy.

- **Decomposition**  
  Simplify model layers using techniques like truncated SVD to reduce computational load while preserving performance.

- **Knowledge Distillation**  
  Optionally transfer knowledge from a larger pre-trained model to a smaller one for further compression benefits.

- **Hybrid Compression Pipeline**  
  Seamlessly combine all supported techniques in one pipeline to achieve unmatched efficiency.

- **Comprehensive Evaluation**  
  Evaluate the original and compressed models using key metrics including accuracy, inference time, and model size.

---

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch & Torchvision
- TIMM (for additional model support)
- Transformers (for Hugging Face models)
- scikit-learn, tensorly, and other dependencies (see `requirements.txt`)

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

   Edit `config.yaml` to set your model parameters, device preference, batch size, learning rate, and number of epochs.

---

## Usage

### Command-Line Interface

QDP Studio is run via `main.py`, which allows you to select the dataset and compression techniques.

**Example Command:**

```bash
python main.py --dataset CIFAR10 --prune --quantize --decompose
```

This command will:
- Train a model (default: ResNet18) on CIFAR10.
- Apply pruning, quantization, and decomposition.
- Evaluate and compare performance across the original and compressed models.

**Available Arguments:**

- `--dataset`: Specify the dataset (CIFAR10, MNIST, ImageNet).
- `--batch_size`: Define the batch size.
- `--prune`: Enable pruning.
- `--quantize`: Enable quantization.
- `--decompose`: Enable decomposition.
- `--all`: Run all compression techniques as a hybrid pipeline.

### Hybrid Compression

Hybrid compression applies all supported techniques sequentially. The process involves:
1. **Training:** Begin with a well-trained base model.
2. **Sequential Compression:**  
   - **Pruning:** Remove redundant weights.  
   - **Quantization:** Convert weights to lower-precision.  
   - **Decomposition:** Simplify model layers.
3. **Fine-Tuning:** Recover any lost accuracy after each compression stage.
4. **Evaluation:** Compare the original and compressed models based on accuracy, inference time, and model size.

Run the hybrid pipeline using:

```bash
python main.py --dataset CIFAR10 --all
```

---

## Citation

If you use QDP Studio in your research or projects, please cite our work:

```bibtex
@ARTICLE{10833607,
  author={Chaudhari, Jay N. and Galiyawala, Hiren and Sharma, Paawan and Shukla, Pancham and Raval, Mehul S.},
  journal={IEEE Access}, 
  title={Onboard Person Retrieval System With Model Compression: A Case Study on Nvidia Jetson Orin AGX}, 
  year={2025},
  volume={13},
  number={},
  pages={8257-8269},
  keywords={Clothing;Image edge detection;Surveillance;Quantization (signal);Real-time systems;Performance evaluation;Image color analysis;Graphics processing units;Videos;Computational modeling;Edge device;model compression;person attribute recognition;person retrieval;pruning;quantization;surveillance},
  doi={10.1109/ACCESS.2025.3527134}
}
```

---

## Acknowledgements

- Supported by the Science, Technology, and Innovation (STI) Policy of Gujarat Council of Science and Technology, Department of Science and Technology, Government of the Gujarat State, India (Grant Number: GUJCOST/STI/2021-22/3858) under the project "Person Retrieval in Video Surveillance."
- Thanks to the PyTorch and Torchvision communities for their outstanding tools and documentation.
- Inspired by various model compression techniques and research in deep learning.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push the branch: `git push origin feature/my-new-feature`
5. Open a pull request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Start compressing your models with QDP Studio and help push the boundaries of efficient deep learning deployments!

---
