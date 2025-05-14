# models/model.py

import torch
import torch.nn as nn

class SharedFeatureExtractor(nn.Module):
    """Neural network module for extracting shared features from input.

    Applies linear transformation, batch normalization, activation, and dropout.

    Args:
        input_size (int): Size of input feature vector.
        hidden_size (int): Number of neurons in the first hidden layer.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size=128, dropout=0.5):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass for feature extraction.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Extracted feature representation.
        """
        return self.feature_net(x)


class DualOutputModel(nn.Module):
    """Model with shared encoder and two heads for multi-task learning.

    Outputs:
        - CTR prediction (regression)
        - Headline quality prediction (binary classification)

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Size of hidden layers.
    """
    def __init__(self, input_size, hidden_size=128):
        super(DualOutputModel, self).__init__()
        self.shared = SharedFeatureExtractor(input_size, hidden_size)

        self.ctr_head = nn.Linear(hidden_size // 2, 1)
        self.quality_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        """Forward pass through shared layers and both output heads.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: (CTR prediction, quality prediction)
        """
        features = self.shared(x)
        return self.ctr_head(features), self.quality_head(features)


class HeadlineClassifier(nn.Module):
    """Simple feed-forward classifier for headline quality classification.

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Size of the hidden layer.
        output_classes (int): Number of output classes (default: 2).
    """
    def __init__(self, input_size, hidden_size=128, output_classes=2):
        super(HeadlineClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_classes)
        )

    def forward(self, x):
        """Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Logits for classification.
        """
        return self.net(x)
