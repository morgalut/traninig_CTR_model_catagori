# main.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    BATCH_SIZE,
    HIDDEN_DIM,
    EPOCHS,
    LEARNING_RATE,
    ALPHA,
    BETA,
    MODEL_SAVE_PATH,
    OUTPUT_PATH,
    TASK_TYPE,
    EARLY_STOPPING_PATIENCE,
    CTR_MIN,
    CTR_MAX,
)

from utils import HeadlinePreprocessor, validate_csv_files
from load_data.data_loader import HeadlineDataset
from models.model import DualOutputModel
from training.loss import HybridLoss
from training.trainer import train_model


def compute_pos_weight(csv_path):
    """Compute the positive class weight for binary classification loss.

    Args:
        csv_path (str): Path to the training CSV file.

    Returns:
        torch.Tensor: A tensor with the computed positive class weight,
                      or None if class distribution is invalid.
    """
    df = pd.read_csv(csv_path)
    preprocessor = HeadlinePreprocessor()
    df["label"] = preprocessor.generate_classification_label(df)
    label_counts = df["label"].value_counts()
    if 0 not in label_counts or 1 not in label_counts:
        return None
    neg = label_counts[0]
    pos = label_counts[1]
    return torch.tensor([neg / pos], dtype=torch.float32)


def inverse_minmax(x, original_min=0.0, original_max=1.0):
    """Apply inverse min-max scaling to convert normalized values back to original range.

    Args:
        x (np.ndarray or float): Scaled value(s).
        original_min (float): Minimum of original range.
        original_max (float): Maximum of original range.

    Returns:
        float or np.ndarray: Value(s) rescaled to original range.
    """
    return x * (original_max - original_min) + original_min


def save_predictions(model, dataloader, test_csv_path, output_path):
    """Run inference on test data and save CTR predictions to a CSV file.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Test data loader.
        test_csv_path (str): Path to test CSV file.
        output_path (str): Path to save the predictions CSV.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_ctr = []
    with torch.no_grad():
        for batch_X in dataloader:
            if isinstance(batch_X, (tuple, list)):
                batch_X = batch_X[0]
            batch_X = batch_X.to(device)
            ctr_pred, _ = model(batch_X)
            ctr_pred = ctr_pred.cpu().squeeze().numpy()

            # ✅ Apply inverse scaling
            ctr_pred = inverse_minmax(ctr_pred, CTR_MIN, CTR_MAX)

            preds_ctr.extend(ctr_pred.tolist())

    df_out = pd.read_csv(test_csv_path).copy()
    df_result = pd.DataFrame({
        "id": df_out["id"],
        "Predicted Normalized URL CTR": preds_ctr
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_csv(output_path, index=False)
    print(f"\n📁 Saved test CTR predictions to: {output_path}")

    # ✅ Optional CTR comparison if ground truth exists
    if "Normalized URL CTR" in df_out.columns:
        plot_ctr_comparison(df_out["Normalized URL CTR"], df_result["Predicted Normalized URL CTR"])


def plot_ctr_comparison(true_ctr, pred_ctr, output_path="data/output/ctr_scatter.png"):
    """Generate a scatter plot comparing predicted vs. true CTR values.

    Args:
        true_ctr (pd.Series): Ground truth CTR values.
        pred_ctr (list or pd.Series): Predicted CTR values.
        output_path (str): File path to save the generated plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(true_ctr, pred_ctr, alpha=0.6)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("True Normalized CTR")
    plt.ylabel("Predicted Normalized CTR")
    plt.title("Predicted vs. True CTR")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"📈 Saved CTR scatter plot to: {output_path}")


def main():
    """Main training and inference pipeline.

    Performs the following steps:
    - Validates input CSV files.
    - Loads and prepares training data.
    - Initializes model and optionally loads from checkpoint.
    - Trains the model using hybrid loss.
    - Loads test data and runs inference.
    - Saves predictions and generates comparison plots.
    """
    print("🔍 Validating CSV files...")
    validate_csv_files(TRAIN_CSV_PATH, TEST_CSV_PATH)

    print("📥 Loading training data...")
    train_dataset = HeadlineDataset(TRAIN_CSV_PATH, training=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = next(iter(train_loader))[0].shape[1]
    pos_weight = compute_pos_weight(TRAIN_CSV_PATH)

    if TASK_TYPE != "dual":
        raise ValueError(f"Unsupported TASK_TYPE: {TASK_TYPE}")

    # ✅ Model init and continuation support
    model = DualOutputModel(input_size=input_dim, hidden_size=HIDDEN_DIM)
    if os.path.exists(MODEL_SAVE_PATH):
        print("🔁 Continuing training from saved model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("🆕 Training from scratch...")

    loss_fn = HybridLoss(alpha=ALPHA, beta=BETA, pos_weight=pos_weight)

    print("🚀 Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path=MODEL_SAVE_PATH,
        alpha=ALPHA,
        beta=BETA,
        verbose=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        pos_weight=pos_weight
    )

    # ✅ Inference after training
    print("📦 Loading trained model for inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

    print("\n📥 Loading test data...")
    test_dataset = HeadlineDataset(TEST_CSV_PATH, training=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("📤 Running inference on test data...")
    save_predictions(model, test_loader, TEST_CSV_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
