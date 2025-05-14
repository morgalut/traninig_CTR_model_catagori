# 🧠 Dual-Output CTR + Quality Classifier Trainer

This module provides a training loop for a **dual-output PyTorch model** that simultaneously performs:

- **CTR Regression** (continuous output)
- **Headline Quality Classification** (binary output)

It uses a custom **Hybrid Loss** that balances both tasks during training.

---

## 📁 File

`training/trainer.py`

---

## 🚀 Features

- 🎯 **Dual-task Training**: Regress Click-Through Rate (CTR) and classify headline quality.
- 🧪 **Validation Metrics**: Includes AUC, F1 score, accuracy, MAE.
- ⏸️ **Early Stopping**: Stops training if validation loss doesn't improve.
- 📉 **Learning Rate Scheduler**: Reduces LR when validation loss plateaus.
- 📊 **Metric Plotting**: Saves training/validation performance plots as PNG.

---

## 🛠️ Key Functions

### `train_model(...)`

Trains a given model with both regression and classification objectives.

**Arguments:**

- `model`: Your PyTorch model with two outputs.
- `train_loader`, `val_loader`: `DataLoader`s for training/validation.
- `epochs`, `lr`: Training hyperparameters.
- `alpha`, `beta`: Control the loss balance (e.g., 0.3 CTR, 0.7 classification).
- `pos_weight`: Weight for positive class (imbalanced classification).
- `save_path`: Where to save the best model weights.
- `early_stopping_patience`: Number of epochs to wait for improvement.

### `evaluate(...)`

Evaluates the model on a validation set and returns:

- Average Loss
- Accuracy
- Mean Absolute Error (MAE)
- ROC AUC Score
- F1 Score (based on optimal threshold)

### `plot_metrics(...)`

Plots and saves:

- Train MAE
- Validation MAE
- F1 score
- AUC

📍 Output: `pt_save/img/ctr_mae_plot.png`

---

## 💾 Output

- **Best model weights** saved at `save_path` you provide.
- **Metrics Plot** at: `pt_save/img/ctr_mae_plot.png`

---

## 📌 Example Usage

```python
from training.trainer import train_model
from model import DualOutputModel  # your model
from data_loader import get_data_loaders  # assumes your function

model = DualOutputModel(...)
train_loader, val_loader = get_data_loaders(...)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=1e-4,
    save_path="pt_save/best_model.pt",
    alpha=0.4,
    beta=0.6,
    early_stopping_patience=3
)
