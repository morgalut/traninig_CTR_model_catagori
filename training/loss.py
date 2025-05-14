# training/loss.py

# Import PyTorch core and neural network modules
import torch.nn as nn


# Custom loss function combining MSE (regression) and BCE (classification)
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, pos_weight=None):
        super(HybridLoss, self).__init__()

        # Weight for regression (CTR) loss
        self.alpha = alpha

        # Weight for classification (headline quality) loss
        self.beta = beta

        # Mean Squared Error loss for CTR prediction
        self.mse = nn.MSELoss()

        # Binary Cross-Entropy loss for quality classification
        # Applies optional positive class weighting to handle class imbalance
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_ctr, true_ctr, pred_quality, true_quality):
        # Compute MSE loss between predicted and true CTR
        ctr_loss = self.mse(pred_ctr, true_ctr)

        # Compute BCE loss between predicted and true quality labels
        class_loss = self.bce(pred_quality, true_quality)

        # Return weighted sum of both losses
        return self.alpha * ctr_loss + self.beta * class_loss
