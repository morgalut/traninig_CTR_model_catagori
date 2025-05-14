# utils.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import config

def validate_csv_files(train_path, test_path):
    """Validate the training and test CSV files.

    Ensures the files can be loaded and contain the expected 'Normalized URL CTR' column.
    Also checks that CTR values in training data are within the range [0, 1].
    Prints a warning if test data lacks CTR values (valid for inference mode).
    """

    import pandas as pd

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load CSV files: {e}")

    # Check required column in training data
    if 'Normalized URL CTR' not in train_df.columns:
        raise ValueError("❌ 'Normalized URL CTR' is missing from training data.")

    # Check CTR range to validate scaling
    ctr_min = train_df['Normalized URL CTR'].min()
    ctr_max = train_df['Normalized URL CTR'].max()

    if not (0 <= ctr_min <= 1 and 0 <= ctr_max <= 1):
        raise ValueError("❌ 'Normalized URL CTR' in training data is not scaled to [0, 1].")

    # Optional: warn if test data lacks CTR (this is expected for inference)
    if 'Normalized URL CTR' not in test_df.columns:
        print("⚠️ 'Normalized URL CTR' is missing from test data. Assuming inference-only mode.")

    print("✅ CSV validation passed.\n")

class HeadlinePreprocessor:
    """Utility class for preprocessing article headline data.

    Includes methods for label generation, categorical encoding, numeric scaling,
    TF-IDF transformation, and dataset splitting for regression/classification tasks.
    """

    def __init__(self, tfidf_max_features=100):
        """Initialize the preprocessor with TF-IDF and scaling tools.

        Args:
            tfidf_max_features (int): Maximum number of features for TF-IDF vectorizer.
        """
        self.vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def generate_classification_label(self, df, ctr_column="Normalized URL CTR", cap_positive=True, max_positive_ratio=0.5):
        """Generate binary labels by comparing CTR values to median.

        Optionally caps the proportion of positive labels to avoid imbalance.

        Args:
            df (pd.DataFrame): Input dataframe with CTR column.
            ctr_column (str): Column name for CTR values.
            cap_positive (bool): Whether to limit positive labels.
            max_positive_ratio (float): Max ratio of positive samples.

        Returns:
            pd.Series: Series of binary labels.
        """
        median_ctr = df[ctr_column].median()
        labels = (df[ctr_column] > median_ctr).astype(int)

        if cap_positive:
            max_pos = int(max_positive_ratio * len(labels))
            current_pos = labels.sum()
            if current_pos > max_pos:
                pos_indices = labels[labels == 1].sample(n=max_pos, random_state=42).index
                labels[:] = 0
                labels.loc[pos_indices] = 1

        return labels

    def encode_categoricals(self, df, columns):
        """Encode categorical features using LabelEncoder.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (list): List of column names to encode.

        Returns:
            pd.DataFrame: Transformed dataframe with encoded columns.
        """
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def fit_transform(self, df, task='regression'):
        """Apply all preprocessing steps and return tensors.

        Steps include:
        - Null filling
        - Dropping unused columns
        - Label generation for classification
        - Categorical encoding
        - TF-IDF transformation
        - Feature scaling

        Args:
            df (pd.DataFrame): Input dataframe.
            task (str): Task type - 'regression' or 'classification'.

        Returns:
            Tuple[Tensor, Tensor or None]: Feature tensor and label tensor or None.
        """
        df = df.copy()
        df = df.fillna('None')

        if 'Landing Page' in df.columns and 'id' in df.columns:
            df = df.drop(columns=['Landing Page', 'id'])

        if task == 'classification':
            df['label'] = self.generate_classification_label(df)

        categorical_cols = ['category_name', 'relationship', 'cluster', 'STR_High_Low', 'is_question']
        df = self.encode_categoricals(df, categorical_cols)

        tfidf_matrix = self.vectorizer.fit_transform(df['post_title']).toarray()

        numeric_cols = [
            'category_name', 'relationship', 'Impressions',
            'word_count', 'character_count', 'contains_number', 'contains_question',
            'contains_quote', 'special_char_count', 'CTR_per_cluster_avg',
            'keyword_strength', 'contains_keywords_score'
        ]
        numeric_data = df[numeric_cols].astype(float)
        numeric_scaled = self.scaler.fit_transform(numeric_data)

        X = np.hstack([numeric_scaled, tfidf_matrix]).astype(np.float32)

        # ✅ Only try to extract labels if present
        if task == 'regression':
            if 'Normalized URL CTR' in df.columns:
                y = df['Normalized URL CTR'].astype(np.float32).values.reshape(-1, 1)
                return torch.tensor(X), torch.tensor(y)
            else:
                return torch.tensor(X), None
        else:
            y = df['label'].astype(np.int64).values
            return torch.tensor(X), torch.tensor(y)

    def split_data(self, X, y, test_size=0.2, seed=42):
        """Split data into training and validation sets with stratification.

        Args:
            X (Tensor): Features tensor.
            y (Tensor): Labels tensor.
            test_size (float): Fraction of data to use for testing.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: X_train, X_test, y_train, y_test.
        """
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    def get_ctr_minmax(train_path):
        """Get the minimum and maximum 'Normalized URL CTR' from CSV file.

        Args:
            train_path (str): Path to the training CSV.

        Returns:
            Tuple[float, float]: Minimum and maximum CTR values.
        """
        df = pd.read_csv(train_path)
        ctr_min = df["Normalized URL CTR"].min()
        ctr_max = df["Normalized URL CTR"].max()
        return ctr_min, ctr_max
