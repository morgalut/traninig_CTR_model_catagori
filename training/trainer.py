# training/trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

from training.loss import HybridLoss


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    save_path,
    alpha=0.3,
    beta=0.7,
    verbose=True,
    early_stopping_patience=5,
    pos_weight=None
):
    """Train a dual-output PyTorch model with hybrid loss for CTR regression and quality classification.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data (can be None).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        save_path (str): Path to save the model weights.
        alpha (float): Weight for CTR loss component.
        beta (float): Weight for quality classification loss.
        verbose (bool): Whether to print training and validation metrics.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        pos_weight (Tensor or None): Positive class weight for classification loss.

    Saves:
        Trained model weights to `save_path`.

    Plots:
        Performance metrics if `val_loader` is provided.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    loss_fn = HybridLoss(alpha=alpha, beta=beta, pos_weight=pos_weight)
    mae_metric = nn.L1Loss()

    train_mae_list, val_mae_list, f1_scores, auc_scores = [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples, total_mae = 0.0, 0, 0, 0.0

        for batch_X, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch_X, batch_targets = batch_X.to(device), batch_targets.to(device)
            true_ctr, true_quality = batch_targets[:, 0].unsqueeze(1), batch_targets[:, 1].unsqueeze(1)

            optimizer.zero_grad()
            pred_ctr, pred_quality = model(batch_X)
            loss = loss_fn(pred_ctr, true_ctr, pred_quality, true_quality)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.sigmoid(pred_quality) > 0.5
            total_correct += (predictions == true_quality.bool()).sum().item()
            total_samples += true_quality.size(0)
            total_mae += mae_metric(pred_ctr, true_ctr).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_correct / total_samples
        avg_train_mae = total_mae / len(train_loader)

        if val_loader is not None:
            val_loss, val_acc, val_mae, val_auc, val_f1 = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)

            train_mae_list.append(avg_train_mae)
            val_mae_list.append(val_mae)
            f1_scores.append(val_f1)
            auc_scores.append(val_auc)

            if verbose:
                print(
                    f"🧠 Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {avg_train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Train MAE: {avg_train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                    f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("⏹️ Early stopping triggered.")
                    break
        else:
            if verbose:
                print(
                    f"🧠 Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Train Acc: {avg_train_acc:.4f} | Train MAE: {avg_train_mae:.4f}"
                )

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")

    if val_loader is not None:
        plot_metrics(train_mae_list, val_mae_list, f1_scores, auc_scores)


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate the model on validation data.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function instance.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (avg_loss, avg_accuracy, avg_mae, auc_score, f1_score)
    """
    model.eval()
    total_loss, total_correct, total_samples, total_mae = 0.0, 0, 0, 0.0
    mae_metric = nn.L1Loss()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_targets in dataloader:
            batch_X, batch_targets = batch_X.to(device), batch_targets.to(device)
            true_ctr, true_quality = batch_targets[:, 0].unsqueeze(1), batch_targets[:, 1].unsqueeze(1)

            pred_ctr, pred_quality = model(batch_X)
            loss = loss_fn(pred_ctr, true_ctr, pred_quality, true_quality)
            total_loss += loss.item()

            predictions = torch.sigmoid(pred_quality) > 0.5
            total_correct += (predictions == true_quality.bool()).sum().item()
            total_samples += true_quality.size(0)
            total_mae += mae_metric(pred_ctr, true_ctr).item()

            all_preds.extend([float(p) for p in torch.sigmoid(pred_quality).cpu().numpy().flatten()])
            all_labels.extend([int(l) for l in true_quality.cpu().numpy().flatten()])

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    avg_mae = total_mae / len(dataloader)

    try:
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores = np.nan_to_num(f1_scores)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            f1 = f1_score(all_labels, [p > optimal_threshold for p in all_preds])
        else:
            auc, f1 = -1, -1
    except Exception as e:
        print(f"⚠️ Metric error: {e}")
        auc, f1 = -1, -1

    return avg_loss, avg_acc, avg_mae, auc, f1


def plot_metrics(train_mae, val_mae, f1_scores, auc_scores):
    """Plot and save training/validation metrics across epochs.

    Args:
        train_mae (list): List of training MAE values.
        val_mae (list): List of validation MAE values.
        f1_scores (list): List of validation F1 scores.
        auc_scores (list): List of validation AUC scores.

    Saves:
        A PNG file visualizing the metrics over training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_mae, label="Train MAE")
    plt.plot(val_mae, label="Val MAE")
    plt.plot(f1_scores, label="Val F1")
    plt.plot(auc_scores, label="Val AUC")
    plt.title("Model Performance Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Error / Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("pt_save/img", exist_ok=True)
    plt.savefig("pt_save/img/ctr_mae_plot.png")
    print("📊 Saved performance plot to pt_save/img/ctr_mae_plot.png")
