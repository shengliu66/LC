""" Test group mixup.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

SPDX-License-Identifier: CC-BY-NC-4.0
"""

import torch

import group_mixup


def test_obtain_groups():
    """Test obtain groups function."""
    label = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)])
    num_classes = 2
    minority_gt = {0: [1], 1: [2]}
    target_gt = {0: [0, 1], 1: [2, 3]}
    group_minority_index, group_target_label = group_mixup.obtain_groups(label, num_classes)

    assert group_minority_index == minority_gt
    assert group_target_label == target_gt


def test_mix_up():
    """Test Mixup."""
    grouped_minority_index = {0: [1], 1: [2]}
    grouped_target_index = {0: [0, 1], 1: [2, 3]}
    feature = torch.rand(4, 6)
    correction = torch.tensor([0.1, 0.4, 0.2, 0.3])
    label = torch.tensor([0, 0, 1, 1])
    tau = 0.1
    output = group_mixup.mixUp(grouped_minority_index, grouped_target_index, feature, label, correction, tau)
    assert output["mixed_feature"].shape == (4, 6)

    # No minority sample for label 0.
    grouped_minority_index = {0: [], 1: [2]}
    output = group_mixup.mixUp(grouped_minority_index, grouped_target_index, feature, label, correction, tau)
    assert output["mixed_feature"].shape == (4, 6)


def test_group_mix_up():
    """Test Group Mixup."""
    label = torch.tensor([0, 0, 1, 1])
    bias_label = torch.tensor([0, 1, 0, 1])
    num_classes = 2
    feature = torch.rand(4, 6)
    correction = torch.tensor([0.1, 0.4, 0.2, 0.3])
    tau = 0.1
    output = group_mixup.group_mixUp(feature, bias_label, correction, label, num_classes, tau)
    assert output["mixed_feature"].shape == (4, 6)


if __name__ == '__main__':
    test_obtain_groups()
    test_mix_up()
    test_group_mix_up()
