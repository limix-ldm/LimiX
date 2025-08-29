import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from torch.utils.data import DistributedSampler


def shuffle_data_along_dim(X: torch.Tensor | np.ndarray, dim: int = 0) -> torch.Tensor | np.ndarray:
    """
        Shuffles data (torch.Tensor or np.ndarray) along a specified axis.

        Args:
            X (torch.Tensor | np.ndarray): The input multidimensional tensor or array.
            dim (int): The dimension along which to shuffle elements.

        Returns:
            X_(torch.Tensor | np.ndarray): A new tensor or array with elements shuffled along the specified dimension.
        """
    if isinstance(X, np.ndarray):
        shuffled_indices = np.random.permutation(X.shape[dim])
        reshaped_indices = shuffled_indices.reshape(
            tuple(1 if i != dim else -1 for i in range(X.ndim))
        )
        shuffled_array = np.take_along_axis(X, reshaped_indices, axis=dim)
        return shuffled_array
    elif isinstance(X, torch.Tensor):
        dim_size = X.size(dim)
        shuffled_indices = torch.randperm(dim_size, device=X.device)
        index_shape = [1] * X.dim()
        index_shape[dim] = dim_size
        expanded_indices = shuffled_indices.view(index_shape)
        broadcasted_indices = expanded_indices.expand_as(X)
        shuffled_tensor = torch.gather(X, dim, broadcasted_indices)
        return shuffled_tensor
    else:
        raise TypeError("Data must be a torch.Tensor or np.ndarray")


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError as e:
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)


def calculate_result(y_test_encoded, y_pred_proba):
    y_pred_label = np.argmax(y_pred_proba, axis=1)
    if len(np.unique(y_test_encoded)) == 2:
        final_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
    else:
        final_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class="ovo")
    print(f"✅ AUC = {final_auc:.4f}")

    # --- Accuracy ---
    acc = accuracy_score(y_test_encoded, y_pred_label)
    print(f"✅ Accuracy = {acc:.4f}")

    # --- F1 Score ---
    f1 = f1_score(y_test_encoded, y_pred_label, average='macro' if len(np.unique(y_test_encoded)) > 2 else 'binary')
    print(f"✅ F1 Score = {f1:.4f}")

    # --- Cross Entropy / LogLoss ---
    ce = log_loss(y_test_encoded, y_pred_proba)
    print(f"✅ LogLoss (Cross Entropy) = {ce:.4f}")

    # --- ECE (Expected Calibration Error) ---
    def compute_ece(y_true, y_prob, n_bins=10):
        """Expected Calibration Error (ECE) implementation"""
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            confidences = np.max(y_prob, axis=1)
            predictions = np.argmax(y_prob, axis=1)
        else:
            confidences = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
            predictions = (confidences >= 0.5).astype(int)

        accuracies = (predictions == y_true)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                avg_conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
        return ece

    ece = compute_ece(y_test_encoded, y_pred_proba, n_bins=10)
    print(f"✅ ECE (Expected Calibration Error, 10 bins) = {ece:.4f}")

    return acc, final_auc, f1, ce, ece





def generate_infenerce_config(args):
    retrieval_config = dict(
        use_retrieval=False,
        retrieval_before_preprocessing=False,
        calculate_feature_attention=False,
        calculate_sample_attention=False,
        subsample_ratio=1,
        subsample_type=None,
        use_type=None,
    )

    config_list = [
        dict(RebalanceFeatureDistribution=dict(worker_tags=["quantile"], discrete_flag=False, original_flag=True,
                                               svd_tag="svd"),
             CategoricalFeatureEncoder=dict(encoding_strategy="ordinal_strict_feature_shuffled"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=["quantile"], discrete_flag=False, original_flag=True,
                                               svd_tag="svd"),
             CategoricalFeatureEncoder=dict(encoding_strategy="ordinal_strict_feature_shuffled"),
             FeatureShuffler=dict(mode="shuffle"), retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=[None], discrete_flag=True, original_flag=False,
                                               svd_tag=None),
             CategoricalFeatureEncoder=dict(encoding_strategy="numeric"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=[None], discrete_flag=True, original_flag=False,
                                               svd_tag=None),
             CategoricalFeatureEncoder=dict(encoding_strategy="numeric"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config)
    ]

    with open(args.inference_config_path, 'w') as f:
        json.dump(config_list, f)


class NonPaddingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_samples = len(range(rank, len(dataset), num_replicas))
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

def swap_rows_back(tensor, indices):
    """

    Args:
        tensor (torch.Tensor):
        indices (list|torch.Tensor):

    Returns:
        torch.Tensor:
    """
    inverse_indices = [0] * len(indices)
    for i, idx in enumerate(indices):
        inverse_indices[idx] = i
    return tensor[inverse_indices]

if __name__ == "__main__":
    args = init_args()
    generate_infenerce_config(args)
