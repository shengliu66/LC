""" Test all loss functions.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

SPDX-License-Identifier: CC-BY-NC-4.0
"""

import torch

import loss


def test_generalized_CE_loss():
    """Test GCE Loss."""
    gce_loss = loss.GeneralizedCELoss()
    logits = torch.tensor([[0.1, 0.3], [0.6, 0.1]])
    labels = torch.tensor([1, 0])
    gce_loss(logits, labels)


def test_logit_correction_loss():
    """Test GCE Loss."""
    lc_loss = loss.LogitCorrectionLoss()
    logits = torch.tensor([[0.1, 0.3], [0.6, 0.1]])
    prior = torch.tensor([[0.2, 0.3], [0.4, 0.1]])
    labels = torch.tensor([1, 0])
    lc_loss(logits, labels, prior)
    lc_loss(logits, labels)


def test_reweighted_CE_loss():
    """Test GCE Loss."""
    w_ce_loss = loss.ReweightedCELoss()
    logits = torch.tensor([[0.1, 0.3], [0.6, 0.1]])
    prior = torch.tensor([[0.2, 0.3], [0.4, 0.1]])
    labels = torch.tensor([1, 0])
    w_ce_loss(logits, labels, prior)


if __name__ == '__main__':
    test_generalized_CE_loss()
    test_logit_correction_loss()
    test_reweighted_CE_loss()
