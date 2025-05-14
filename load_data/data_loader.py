# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import LABEL_NOISE_PROB, SMOOTHING
from utils import HeadlinePreprocessor
import random

class HeadlineDataset(Dataset):
    """Custom PyTorch dataset for loading and preprocessing headline data.

    Handles both training and inference modes. For training, it applies preprocessing,
    classification label generation, label smoothing, and optional label noise.

    Args:
        csv_file (str): Path to CSV file containing data.
        tfidf_max_features (int): Number of features to use in TF-IDF vectorization.
        training (bool): Whether to load labels (True) or just features (False).
    """
    def __init__(self, csv_file, tfidf_max_features=100, training=True):
        self.df = pd.read_csv(csv_file)
        self.training = training

        # ✅ Only raise error in training mode
        if self.training and 'Normalized URL CTR' not in self.df.columns:
            raise ValueError("Missing 'Normalized URL CTR'. You must preprocess with log1p + min-max scaling.")

        # Initialize preprocessor
        self.preprocessor = HeadlinePreprocessor(tfidf_max_features=tfidf_max_features)

        # Generate input features
        self.X, _ = self.preprocessor.fit_transform(self.df, task='regression')

        # In training mode, set targets
        if self.training:
            self.df['label'] = self.preprocessor.generate_classification_label(
                self.df, ctr_column="Normalized URL CTR", cap_positive=True, max_positive_ratio=0.5
            )
            self.ctr_targets = self.df['Normalized URL CTR'].astype('float32').values
            self.label_targets = self.df['label'].astype('float32').values

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a single data sample by index.

        Returns:
            If training:
                Tuple[Tensor, Tensor]: (input_features, target_values)
            If not training:
                Tensor: input_features only
        """
        x = self.X[idx]

        # If test mode, return only features
        if not self.training:
            return x

        ctr = self.ctr_targets[idx]
        label = self.label_targets[idx]

        # Apply label smoothing
        label = (1.0 - SMOOTHING) * label + (SMOOTHING * (1 - label))

        # Inject label noise with configured probability
        if random.random() < LABEL_NOISE_PROB:
            label = 1.0 - label

        y = torch.tensor([ctr, label], dtype=torch.float32)
        return x, y


def get_dataloader(csv_path, batch_size=32, shuffle=True, training=True):
    """Create a PyTorch DataLoader for the HeadlineDataset.

    Args:
        csv_path (str): Path to the CSV file.
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the data.
        training (bool): Whether to load targets (True) or just features (False).

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = HeadlineDataset(csv_file=csv_path, training=training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
