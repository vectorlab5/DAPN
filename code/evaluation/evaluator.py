"""
Evaluation module for DAPN.

Provides comprehensive evaluation including:
- Standard few-shot evaluation
- Cross-domain evaluation
- Ablation studies
- Statistical significance testing
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

from models import DAPN, build_dapn
from data import get_dataset, EpisodeSampler, CrossDomainEpisodeSampler
from configs.config import ExperimentConfig
from utils.metrics import AverageMeter, accuracy, compute_confidence_interval


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for DAPN few-shot classification.
    
    Supports multiple evaluation modes:
    - Standard: support and query from same domain distribution
    - Cross-domain: support from one domain, query from another
    - Ablation: evaluate variants with components disabled
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: DAPN,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.model = model
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        test_dataset,
        n_episodes: int = 600,
        cross_domain: bool = False
    ) -> Dict[str, float]:
        """
        Run standard evaluation.
        
        Args:
            test_dataset: Test dataset
            n_episodes: Number of test episodes
            cross_domain: Whether to use cross-domain evaluation
            
        Returns:
            Dictionary with accuracy, std, confidence interval
        """
        cfg = self.config.training
        n_support = cfg.n_way * cfg.k_shot
        n_query = cfg.n_way * cfg.n_query
        
        # Choose appropriate sampler
        if cross_domain:
            sampler = CrossDomainEpisodeSampler(
                labels=test_dataset.get_labels(),
                domain_labels=test_dataset.get_domain_labels(),
                n_way=cfg.n_way,
                k_shot=cfg.k_shot,
                n_query=cfg.n_query,
                episodes_per_epoch=n_episodes
            )
        else:
            sampler = EpisodeSampler(
                labels=test_dataset.get_labels(),
                n_way=cfg.n_way,
                k_shot=cfg.k_shot,
                n_query=cfg.n_query,
                episodes_per_epoch=n_episodes
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=sampler,
            num_workers=self.config.training.num_workers
        )
        
        accuracies = []
        
        with torch.no_grad():
            for images, labels, domains in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Split into support/query
                support_images = images[:n_support]
                query_images = images[n_support:n_support + n_query]
                
                # Relative labeling
                support_labels_abs = labels[:n_support]
                query_labels_abs = labels[n_support:n_support + n_query]
                
                unique_classes = support_labels_abs.unique()
                label_mapping = {c.item(): i for i, c in enumerate(unique_classes)}
                
                support_labels = torch.tensor(
                    [label_mapping[l.item()] for l in support_labels_abs],
                    device=self.device
                )
                query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels_abs],
                    device=self.device
                )
                
                # Forward pass
                outputs = self.model(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    n_way=cfg.n_way,
                    use_graph_refinement=True
                )
                
                # Compute accuracy
                preds = outputs['log_probs'].argmax(dim=1)
                acc = accuracy(preds, query_labels)
                accuracies.append(acc)
        
        # Compute statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        ci95 = 1.96 * std_acc / np.sqrt(len(accuracies))
        
        return {
            'accuracy': mean_acc,
            'std': std_acc,
            'ci95': ci95,
            'n_episodes': len(accuracies)
        }
    
    def evaluate_ablation(
        self,
        test_dataset,
        n_episodes: int = 600
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study by disabling components.
        
        Returns results for:
        - full: Complete model
        - no_domain_adapt: Without adversarial domain adaptation
        - no_disentangle: Without feature disentanglement
        - no_gnn: Without graph-based refinement
        - baseline: Vanilla prototype network
        """
        cfg = self.config.training
        n_support = cfg.n_way * cfg.k_shot
        n_query = cfg.n_way * cfg.n_query
        
        sampler = EpisodeSampler(
            labels=test_dataset.get_labels(),
            n_way=cfg.n_way,
            k_shot=cfg.k_shot,
            n_query=cfg.n_query,
            episodes_per_epoch=n_episodes
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=sampler,
            num_workers=self.config.training.num_workers
        )
        
        # Store accuracies for each variant
        results = {
            'full': [],
            'no_gnn': [],
            'no_disentangle': [],
            'baseline': [],
        }
        
        with torch.no_grad():
            for images, labels, domains in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                support_images = images[:n_support]
                query_images = images[n_support:n_support + n_query]
                
                support_labels_abs = labels[:n_support]
                query_labels_abs = labels[n_support:n_support + n_query]
                
                unique_classes = support_labels_abs.unique()
                label_mapping = {c.item(): i for i, c in enumerate(unique_classes)}
                
                support_labels = torch.tensor(
                    [label_mapping[l.item()] for l in support_labels_abs],
                    device=self.device
                )
                query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels_abs],
                    device=self.device
                )
                
                # Full model
                outputs_full = self.model(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    n_way=cfg.n_way,
                    use_graph_refinement=True
                )
                preds = outputs_full['log_probs'].argmax(dim=1)
                results['full'].append(accuracy(preds, query_labels))
                
                # Without GNN refinement
                outputs_no_gnn = self.model(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    n_way=cfg.n_way,
                    use_graph_refinement=False
                )
                preds = outputs_no_gnn['log_probs'].argmax(dim=1)
                results['no_gnn'].append(accuracy(preds, query_labels))
                
                # Extract raw backbone features for test-time ablation of disentanglement
                support_raw, _, _ = self.model.extract_features(support_images)
                query_raw, _, _ = self.model.extract_features(query_images)
                
                # Compute prototypes on raw backbone features
                raw_prototypes = self.model.compute_prototypes(support_raw, support_labels, cfg.n_way)
                
                # Without feature disentanglement (uses raw features + GNN refinement)
                raw_prototypes_refined, _ = self.model.prototype_refiner(raw_prototypes)
                log_probs_no_dis = self.model.classify(query_raw, raw_prototypes_refined)
                preds = log_probs_no_dis.argmax(dim=1)
                results['no_disentangle'].append(accuracy(preds, query_labels))
                
                # Baseline (uses raw features + no GNN refinement)
                log_probs_baseline = self.model.classify(query_raw, raw_prototypes)
                preds = log_probs_baseline.argmax(dim=1)
                results['baseline'].append(accuracy(preds, query_labels))
        
        # Compute statistics for each variant
        ablation_results = {}
        for variant, accs in results.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            ci95 = 1.96 * std_acc / np.sqrt(len(accs))
            
            ablation_results[variant] = {
                'accuracy': mean_acc,
                'std': std_acc,
                'ci95': ci95
            }
        
        # Compute p-values for ablations vs full model (paired t-test)
        full_accs = np.array(results['full'])
        for variant in results:
            if variant != 'full':
                variant_accs = np.array(results[variant])
                t_stat, p_value = stats.ttest_rel(full_accs, variant_accs)
                ablation_results[variant]['p_value'] = p_value
        
        return ablation_results
    
    def evaluate_disentanglement(
        self,
        test_dataset,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate quality of feature disentanglement.
        
        Trains linear probes to predict:
        - Domain from invariant features (should be near chance)
        - Class from specific features (should be low)
        
        Good disentanglement means:
        - Invariant features contain class info, not domain info
        - Specific features contain domain info, not class info
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Extract features
        inv_features = []
        spec_features = []
        class_labels = []
        domain_labels = []
        
        loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4
        )
        
        with torch.no_grad():
            collected = 0
            for images, labels, domains in loader:
                if collected >= n_samples:
                    break
                
                images = images.to(self.device)
                
                # Extract features
                _, phi_inv, phi_spec = self.model.extract_features(images)
                
                inv_features.append(phi_inv.cpu().numpy())
                spec_features.append(phi_spec.cpu().numpy())
                class_labels.extend(labels.numpy())
                domain_labels.extend(domains.numpy())
                
                collected += len(images)
        
        inv_features = np.concatenate(inv_features)[:n_samples]
        spec_features = np.concatenate(spec_features)[:n_samples]
        class_labels = np.array(class_labels)[:n_samples]
        domain_labels = np.array(domain_labels)[:n_samples]
        
        # Train probes
        results = {}
        
        # Domain classification using invariant features (should be near chance)
        X_train, X_test, y_train, y_test = train_test_split(
            inv_features, domain_labels, test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        results['domain_from_inv'] = clf.score(X_test, y_test)
        
        # Domain classification using specific features (should be high)
        X_train, X_test, y_train, y_test = train_test_split(
            spec_features, domain_labels, test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        results['domain_from_spec'] = clf.score(X_test, y_test)
        
        # Class classification using invariant features (should be high)
        X_train, X_test, y_train, y_test = train_test_split(
            inv_features, class_labels, test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        results['class_from_inv'] = clf.score(X_test, y_test)
        
        # Class classification using specific features (should be low)
        X_train, X_test, y_train, y_test = train_test_split(
            spec_features, class_labels, test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        results['class_from_spec'] = clf.score(X_test, y_test)
        
        return results


def evaluate_dapn(
    config: ExperimentConfig,
    checkpoint_path: str
) -> Dict[str, any]:
    """
    Entry point for evaluating DAPN.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Comprehensive evaluation results
    """
    # Load model
    model = build_dapn(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(config, model)
    
    # Load test dataset
    test_dataset = get_dataset(
        config.data.dataset,
        config.data.data_root,
        split='test',
        image_size=config.data.image_size,
        use_augmentation=False
    )
    
    results = {}
    
    # Standard evaluation
    logger.info("Running standard evaluation...")
    results['standard'] = evaluator.evaluate(
        test_dataset,
        n_episodes=config.evaluation.n_episodes
    )
    logger.info(f"Standard: {results['standard']['accuracy']:.2%} ± {results['standard']['ci95']:.2%}")
    
    # Cross-domain evaluation
    if config.evaluation.cross_domain_eval:
        logger.info("Running cross-domain evaluation...")
        results['cross_domain'] = evaluator.evaluate(
            test_dataset,
            n_episodes=config.evaluation.n_episodes,
            cross_domain=True
        )
        logger.info(f"Cross-domain: {results['cross_domain']['accuracy']:.2%} ± {results['cross_domain']['ci95']:.2%}")
    
    # Ablation study
    logger.info("Running ablation study...")
    results['ablation'] = evaluator.evaluate_ablation(test_dataset)
    for variant, metrics in results['ablation'].items():
        logger.info(f"  {variant}: {metrics['accuracy']:.2%}")
    
    # Disentanglement validation
    logger.info("Running disentanglement validation...")
    results['disentanglement'] = evaluator.evaluate_disentanglement(test_dataset)
    logger.info(f"  Domain from inv: {results['disentanglement']['domain_from_inv']:.2%}")
    logger.info(f"  Domain from spec: {results['disentanglement']['domain_from_spec']:.2%}")
    
    return results
