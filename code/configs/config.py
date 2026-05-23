"""
Configuration management for Domain-Adaptive Prototype Networks.

This module provides a centralized configuration system using dataclasses,
making it easy to modify hyperparameters and experimental settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture configuration for the DAPN model."""
    
    # Feature encoder (ResNet-12 backbone)
    backbone: str = "resnet12"
    feature_dim: int = 512
    
    # Feature disentanglement dimensions
    # The invariant and specific features should sum to feature_dim
    invariant_dim: int = 256
    specific_dim: int = 256
    
    # Domain discriminator
    discriminator_hidden: int = 256
    num_domains: int = 4  # Number of domains in training set
    
    # Graph Neural Network for prototype refinement
    gnn_hidden: int = 256
    gnn_layers: int = 2
    graph_threshold: float = 0.5  # Cosine similarity threshold for edge construction
    
    # Dropout for regularization
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Meta-learning episode configuration
    n_way: int = 5  # Number of classes per episode
    k_shot: int = 1  # Number of support examples per class
    n_query: int = 15  # Number of query examples per class
    
    # Training schedule
    epochs: int = 100
    episodes_per_epoch: int = 100
    batch_size: int = 4  # Number of episodes per batch
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_decay_epochs: List[int] = field(default_factory=lambda: [60, 80])
    lr_decay_factor: float = 0.1
    
    # Loss weights - these balance different objectives
    # Based on the paper's hyperparameter analysis
    lambda_adv: float = 0.5  # Adversarial domain adaptation weight
    lambda_dis: float = 0.3  # Feature disentanglement weight
    lambda_recon: float = 0.2  # Reconstruction weight
    alpha_graph: float = 0.1  # Graph refinement weight
    
    # Gradient reversal layer scaling
    grl_lambda: float = 1.0
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device configuration
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    
    n_episodes: int = 600  # Number of test episodes
    confidence_level: float = 0.95  # For confidence interval computation
    
    # Cross-domain evaluation settings
    cross_domain_eval: bool = True
    
    # Settings for ablation studies
    ablation_components: List[str] = field(default_factory=lambda: [
        "full",
        "no_domain_adapt",
        "no_disentangle", 
        "no_gnn",
        "baseline"
    ])


@dataclass  
class DataConfig:
    """Dataset configuration."""
    
    dataset: str = "ich_benchmark"  # ich_benchmark, mini_imagenet, tiered_imagenet
    data_root: str = "./data"
    
    # Image preprocessing
    image_size: int = 84
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Data augmentation during training
    use_augmentation: bool = True
    
    # Domain labels - these define the domain splits
    # For ICH-Benchmark: regional domains
    domain_names: List[str] = field(default_factory=lambda: [
        "northern", "southern", "eastern", "western"
    ])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str = "dapn_experiment"
    output_dir: str = "./outputs"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 10
    resume_from: Optional[str] = None
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dictionaries
        config_dict = {
            "name": self.name,
            "output_dir": self.output_dir,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "data": self.data.__dict__,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "resume_from": self.resume_from
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            name=config_dict.get("name", "dapn_experiment"),
            output_dir=config_dict.get("output_dir", "./outputs"),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            data=DataConfig(**config_dict.get("data", {})),
            log_interval=config_dict.get("log_interval", 10),
            save_interval=config_dict.get("save_interval", 10),
            resume_from=config_dict.get("resume_from", None)
        )


def get_default_config() -> ExperimentConfig:
    """Return a default configuration for quick experimentation."""
    return ExperimentConfig()


def get_5way_1shot_config() -> ExperimentConfig:
    """Configuration for 5-way 1-shot experiments."""
    config = ExperimentConfig(name="5way_1shot")
    config.training.n_way = 5
    config.training.k_shot = 1
    return config


def get_5way_5shot_config() -> ExperimentConfig:
    """Configuration for 5-way 5-shot experiments."""
    config = ExperimentConfig(name="5way_5shot")
    config.training.n_way = 5
    config.training.k_shot = 5
    return config
