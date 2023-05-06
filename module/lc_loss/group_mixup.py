"""Group MixUp Implimentation.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

SPDX-License-Identifier: CC-BY-NC-4.0
"""

import random
from typing import Dict, List

import numpy as np
import torch


def obtain_groups(merged_labels: List, num_classes: int):
    """ Get index for each label group and index for minority samples in each label group.

    Args:
        merged_labels (List): a list of pairs the first entry is the target label
         and the second entry is the label from the biased classifier.
        num_classes (int): number of classes.

    Returns:
        grouped_target_index (dict): a dictionary with key as the target label and value as the index of the samples.
        grouped_minority_index (dict): a dictionary with key as
            the target label and value as the index of the minority samples.
    """
    grouped_target_index = {}
    grouped_minority_index = {}
    for k in range(0, num_classes):
        grouped_minority_index[k] = []
        grouped_target_index[k] = []
    for i in range(0, len(merged_labels)):
        # Minority samples are with different target and biased prediction.
        if merged_labels[i][0] != merged_labels[i][1]:
            grouped_minority_index[merged_labels[i][0].item()].append(i)
        grouped_target_index[merged_labels[i][0].item()].append(i)
    return grouped_minority_index, grouped_target_index


def group_mixUp(feature: torch.Tensor, bias_label: torch.Tensor, correction: torch.Tensor, label: torch.Tensor,
                num_classes: int, tau: float):
    """Calculate Group mixUp for a batch of samples.

    Args:
        feature (Tensor): a matrix with feature from N samples.
        bias_label (Tensor): prediction from the biased classifier.
        correction (Tensor): the logic correction matrix.
        label (Tensor): the target label of the samples.
        num_classes (int): number of classes.
        tau (float): mixUp parameter in Algorithm 1 in the paper.

    Returns: A dict with following fields
        mixed_feature (Tensor): Mixed feature.
        mixed_correction (Tensor): correction term for the mixUp samples.
        label_majority (Tensor): target label for the majority sample.
        label_minority (Tensor): target label of the minority sample.
        lam (float): the mix ratio.
    """
    merged_target = [(label[i], bias_label[i]) for i in range(0, len(label))]
    target_groups_a, target_groups_b = obtain_groups(merged_target, num_classes)
    return mixUp(target_groups_a, target_groups_b, feature, label, correction, tau)


def mixUp(grouped_minority_index: Dict, grouped_target_index: Dict, feature: torch.Tensor, label: torch.Tensor,
          correction: torch.Tensor, tau: float = 0.5):
    """Calculate mixUp for a batch of samples.

    Args:
        grouped_minority_index (dict): a dictionary with key as the target label and value as the index of the samples.
        grouped_target_index (dict): a dictionary with key as the target label and value as the index of the samples.
        feature (Tensor): a matrix with feature from N samples.
        label (Tensor): the target label of the samples.
        correction (Tensor): the logic correction matrix.
        tau (float): the ratio of the number of samples in each group.

    Returns: A dict with following fields
        mixed_feature (Tensor): Mixed feature.
        mixed_correction (Tensor): correction term for the mixUp samples.
        label_majority (Tensor): target label for the majority sample.
        label_minority (Tensor): target label of the minority sample.
        lam (float): the mix ratio.
    """
    # Get mixed up parameter.
    lam = np.random.uniform(1 - 2 * tau, 1 - tau)
    indices_all_groups = []
    random_indices_all_groups = []
    for k in grouped_target_index.keys():
        indices = grouped_target_index[k]
        indices_all_groups += indices

        if grouped_minority_index[k]:
            # Get minority index.
            draw_indices = torch.randint(len(grouped_minority_index[k]), size=(len(indices),))
            random_indices_all_groups += [grouped_minority_index[k][l] for l in draw_indices]
        else:
            # if no minority index, do regular mixup.
            random_indices_all_groups += random.sample(indices, len(indices))

    indices_all_groups = torch.tensor(indices_all_groups)
    random_indices_all_groups = torch.tensor(random_indices_all_groups)

    # Define return values.
    mixed_feature = None
    mixed_correction = None
    label_majority, label_minority = None, None
    if random_indices_all_groups.nelement() > 0:
        # Mix feature.
        mixed_feature = lam * feature[indices_all_groups] + (1 - lam) * feature[random_indices_all_groups]
        # Mix correction value.
        mixed_correction = lam * correction[indices_all_groups] + (1 - lam) * correction[random_indices_all_groups]
        label_majority = label[indices_all_groups]
        label_minority = label[random_indices_all_groups]

    return {"mixed_feature": mixed_feature, "mixed_correction": mixed_correction, "label_majority": label_majority,
            "label_minority": label_minority, "lam": lam}
