# Domain-Adaptive Prototype Networks (DAPN)

Implementation of Domain-Adaptive Prototype Networks for Few-Shot Intangible Cultural Heritage Image Classification.

## Overview

DAPN is a meta-learning framework that addresses domain shift in few-shot learning by:
1. Learning domain-invariant prototypes through feature disentanglement
2. Using adversarial training to enforce domain-invariance
3. Refining prototypes with Graph Neural Networks based on category relationships

## Project Structure

```
code/
├── models/              # Model definitions
│   ├── feature_encoder.py      # ResNet-12 backbone
│   ├── disentanglement.py      # Feature disentanglement modules
│   ├── gnn_refinement.py       # GNN for prototype refinement
│   ├── dapn.py                 # Complete DAPN model
│   └── __init__.py
├── utils/               # Utilities
│   ├── losses.py               # Loss functions
│   └── __init__.py
├── data/                # Data loaders
│   ├── ich_dataset.py          # ICH dataset and episode sampler
│   └── __init__.py
├── scripts/             # Training and evaluation scripts
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the following structure:
```
data/
├── train/
│   └── class_name/
│       └── domain_name/
│           └── images...
├── val/
│   └── ...
└── test/
    └── ...
```

## Usage

### Training

Train DAPN on the ICH benchmark:

```bash
python scripts/train.py \
    --data_dir /path/to/data \
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

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir /path/to/test/data \
    --n_way 5 \
    --k_shot 1 \
    --n_query 15 \
    --num_episodes 600
```

## Model Architecture

- **Feature Encoder**: ResNet-12 backbone (512-dim features)
- **Disentanglement**: Two-branch network (256-dim invariant + 256-dim specific)
- **Adversarial Discriminator**: Domain classifier (2-layer MLP)
- **Domain Reconstructor**: Domain predictor from concatenated features
- **GNN Refinement**: 2-layer Graph Convolutional Network

## Loss Functions

- **Task Loss**: Cross-entropy for classification
- **Adversarial Loss**: Domain discrimination (with gradient reversal)
- **Orthogonality Loss**: Enforces independence of invariant/specific features
- **Reconstruction Loss**: Ensures domain info is captured in specific features
- **Graph Loss**: Smoothness regularization on refined prototypes

## Hyperparameters

Default hyperparameters (from paper):
- Learning rate: 0.0001
- Weight decay: 1e-4
- Lambda_adv: 0.5
- Lambda_ortho: 0.3
- Lambda_recon: 0.2
- Lambda_graph: 0.1
- Batch size: 16 episodes
- Epochs: 100

## Citation

If you use this code, please cite:

```bibtex
@article{dapn2025,
  title={Domain-Adaptive Prototype Networks for Few-Shot Intangible Cultural Heritage Image Classification},
  author={...},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
```

## Generating Qualitative Visualizations

To generate the qualitative examples figure with real data:

1. **Prepare dataset** (see `scripts/README_QUALITATIVE.md` for details)
2. **Train the model** (as described above)
3. **Generate visualization:**
```bash
python scripts/generate_qualitative_visualization.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir /path/to/data \
    --output ../qualitative_examples_real.pdf
```

For detailed instructions, see `scripts/README_QUALITATIVE.md`.

## License

[Your License Here]
