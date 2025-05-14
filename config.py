# config.py

# Training parameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
DROPOUT = 0.5
ALPHA = 0.3  # CTR loss weight
BETA = 0.7   # Classification loss weight
CTR_MIN = 0.0
CTR_MAX = 1.0
# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3

# Data paths
TRAIN_CSV_PATH = "data/Train_CSV.csv"
TEST_CSV_PATH = "data/Test_CSV_For_Inference.csv"

MODEL_SAVE_PATH = "pt_save/headline_quality_model.pth"
OUTPUT_PATH = "data/output/predictions.csv"

# Task control
RANDOM_SEED = 42
TASK_TYPE = "dual"  # Options: "dual" or "classification"

# Label processing
LABEL_NOISE_PROB = 0.01
SMOOTHING = 0.01
CTR_LABEL_THRESHOLD = 0.01
