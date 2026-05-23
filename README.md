# Domain-Adaptive Prototype Networks (DAPN)

This repository contains the official PyTorch implementation and benchmark data for **Domain-Adaptive Prototype Networks for Few-Shot Intangible Cultural Heritage Image Classification**.

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

Intangible Cultural Heritage (ICH) artifact cataloging faces severe challenges due to data scarcity and regional/material variations (domain shift). **Domain-Adaptive Prototype Networks (DAPN)** addresses these challenges using a meta-learning framework that:
1. **Disentangles** images into domain-invariant content/morphological features and domain-specific style/material features.
2. **Aligns** domain-invariant features using an adversarial domain discriminator with inverse-frequency weighting.
3. **Refines** prototypes globally using a Graph Convolutional Network (GCN) that leverages category-to-category relationships.

For a detailed description of the model and experimental results, please refer to our paper.

---

## 📂 Project Structure

```
DAPN/
├── code/                    # Source Code
│   ├── models/              # Neural Network Architectures
│   │   ├── feature_encoder.py  # ResNet-12 Backbone
│   │   ├── disentanglement.py  # Content-Style Disentanglement Module
│   │   ├── gnn_refinement.py   # GNN-based Prototype Refiner
│   │   └── dapn.py             # Integrated DAPN Model
│   ├── data/                # Data Loading and Sampling
│   │   ├── ich_dataset.py      # ICH Dataset and Episode Sampler
│   │   ├── datasets.py         # Standard Few-Shot Datasets
│   │   └── sampler.py          # Meta-learning Batch Sampler
│   ├── training/            # Model Training logic
│   │   └── trainer.py          # Domain-Adaptive Meta-Trainer
│   ├── evaluation/          # Evaluation pipelines
│   │   └── evaluator.py        # Ablation and Generalization Evaluator
│   ├── scripts/             # Entry point execution scripts
│   │   ├── train.py            # Model training entry script
│   │   ├── evaluate.py         # Evaluation entry script
│   │   └── quick_train.py      # Quick training on synthetic dataset
│   ├── requirements.txt     # Python Dependencies
│   └── README.md            # Code documentation
├── data/                    # Dataset benchmarks
│   ├── ich_benchmark/       # ICH domain splits (Jingdezhen, NCU, etc.)
│   ├── museum_downloads/    # Pre-processed museum collections
│   └── real_datasets/       # Additional domain evaluation splits
└── .gitignore               # Clean repository ignores (excludes paper documents)
```

---

## ⚡ Quick Start

### 1. Installation

Create a virtual environment and install the required packages:

```bash
# Clone the repository
git clone git@github.com:vectorlab5/DAPN.git
cd DAPN

# Install dependencies
pip install -r code/requirements.txt
```

### 2. Prepare Data
Ensure the benchmarks in the `data/` directory are unzipped and formatted correctly. The dataset should follow an episode-friendly layout containing `support` and `query` splits for each domain.

### 3. Training
To train the DAPN model on the ICH-Benchmark:

```bash
cd code
python scripts/train.py \
    --data_dir ../data/ich_benchmark \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 16 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 15 \
    --num_domains 4 \
    --lambda_adv 0.5 \
    --lambda_ortho 0.3 \
    --lambda_recon 0.2 \
    --lambda_graph 0.1
```

### 4. Evaluation
Evaluate your trained checkpoint on the target test domain:

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ../data/ich_benchmark \
    --n_way 5 \
    --k_shot 1 \
    --n_query 15 \
    --num_episodes 600
```

---

## 🔬 Core Components

* **Content-Style Disentanglement**: Separates feature representation into two distinct subspaces: $\mathbf{f}_{inv}$ (invariant, containing category morphology) and $\mathbf{f}_{spec}$ (specific, containing domain details). Orthogonality loss $\mathcal{L}_{ortho}$ ensures the two branches learn independent representations.
* **Inverse-Frequency Adversarial Weighting**: Domain discriminators align $\mathbf{f}_{inv}$ across domains. Class-domain inverse frequency weighting balances discriminator gradients, preventing dominant domain-class pairs from biasing the encoder.
* **Graph Regularization Loss**: Applies a Dirichlet energy loss over the category prototypes to smooth prototypes according to their semantic relationships:
  $$\mathcal{L}_{graph} = \sum_{i < j} A_{ij} \|\mathbf{p}_i - \mathbf{p}_j\|^2$$

---

## 📄 Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{dapn2025,
  title={Domain-Adaptive Prototype Networks for Few-Shot Intangible Cultural Heritage Image Classification},
  author={Nie, Yan and Xu, Guanghao and Qiu, Jing and Wang, Yunqian},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
