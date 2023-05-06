"""Define loss functions.

Including Generalized CE Loss (for the biased branch) and the proposed LC Loss (for the target branch).

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

SPDX-License-Identifier: CC-BY-NC-4.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCELoss(nn.Module):
    """ Generalized Cross Entropy Loss."""

    def __init__(self, q: float = 0.7):
        super(GeneralizedCELoss, self).__init__()
        # q is the hyperparameter for the GCE loss
        self.q = q

    def forward(self, logits, targets):
        """Calculate Loss."""
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise RuntimeError('GCE p is None.')

        y_g = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (y_g.squeeze().detach() ** self.q) * self.q
        if np.isnan(y_g.mean().item()):
            raise RuntimeError('GCE y_g is None')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


class ReweightedCELoss(nn.Module):
    """Define the reweighted CE loss."""

    def forward(self, output, label, biased_prediction):
        weight = biased_prediction[range(len(label)), label]
        loss = 1.0 / weight * F.cross_entropy(output, label, reduction='none')
        return loss


class LogitCorrectionLoss(nn.Module):
    """Define the proposed logit correction loss."""

    def __init__(self, eta: float = 1.):
        super(LogitCorrectionLoss, self).__init__()
        # eta is the hyperparameter for GroupMixUp.
        self.eta = eta

    def forward(self, logits: torch.tensor, label: torch.tensor, prior=torch.tensor(1.)):
        """Calculate Loss."""
        # Calculate the correction.
        correction = torch.log((prior ** self.eta) + 1e-4)
        # add correction to the original logit.
        corrected_logits = logits + correction
        loss = F.cross_entropy(corrected_logits, label, reduction='none')
        return loss
