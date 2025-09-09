import os
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
import csv
from torch.utils.tensorboard import SummaryWriter
import joblib

from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode
from csnn_framework.nn_model import NNModel

# --- Configuration ---
RUN_ID = None  # "13113319" to resume a run

# Base Directories
LOGS_BASE_DIR = "logs/csnn"
MODELS_BASE_DIR = "models/csnn"
DATASET_DIR = "csnn_framework/datasets"

# Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_THRESHOLD = 0.01
MAX_EPOCHS = 200
PATIENCE_LIMIT = 30
DT = 0.001

# Reproducibility
SEED = 44
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_and_preprocess_dataset(controller_name, dataset_dir):
    """Load and preprocess dataset for a controller."""
    filepath = os.path.join(dataset_dir, f"{controller_name.lower()}.csv")
    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at {filepath}")
        return None, None, None

    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Basic dataset statistics
    print(f"  Loaded {len(df)} data points")
    print(f"  Stable trajectories: {df['stable'].sum()}")
    print(f"  Unstable trajectories: {len(df) - df['stable'].sum()}")
    print(f"  Cost-to-go range: [{df['cost_to_go'].min():.3f}, {df['cost_to_go'].max():.3f}]")

    # Extract features and targets
    states = df[['x_k', 'dx_k', 'a_k', 'da_k']].values.astype(np.float32)
    targets = df['cost_to_go'].values.astype(np.float32)

    return states, targets, df


def create_normalized_datasets(states, targets, test_size=0.3, random_state=SEED):
    """Create normalized training and validation datasets."""
    # Train-validation split
    train_states, val_states, train_targets, val_targets = train_test_split(states, targets, test_size=test_size, random_state=random_state)

    # Normalize states
    state_scaler = StandardScaler()
    train_states_norm = state_scaler.fit_transform(train_states)
    val_states_norm = state_scaler.transform(val_states)

    # Normalize targets (CRITICAL for large cost values)
    target_scaler = StandardScaler()
    train_targets_norm = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
    val_targets_norm = target_scaler.transform(val_targets.reshape(-1, 1)).flatten()

    print(f"  State normalization: mean={state_scaler.mean_}, scale={state_scaler.scale_}")
    print(f"  Target normalization: mean={target_scaler.mean_[0]:.3f}, scale={target_scaler.scale_[0]:.3f}")

    # Create Tensor datasets
    train_dataset = TensorDataset(torch.tensor(train_states_norm, dtype=torch.float32), torch.tensor(train_targets_norm, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(val_states_norm, dtype=torch.float32), torch.tensor(val_targets_norm, dtype=torch.float32).view(-1, 1))

    scalers = {'state_scaler': state_scaler, 'target_scaler': target_scaler}

    return train_dataset, val_dataset, scalers


def train_one_epoch(model, loader, optimizer, loss_function, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for states, targets in loader:
        states, targets = states.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions = model(states)
        loss = loss_function(predictions, targets)
        loss.backward()

        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def validate_model(model, loader, loss_function, device):
    """Validate model performance."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for states, targets in loader:
            states, targets = states.to(device), targets.to(device)
            predictions = model(states)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def calculate_rmse(mse_loss):
    """Calculate RMSE from MSE loss."""
    return torch.sqrt(torch.tensor(mse_loss)).item()


if __name__ == '__main__':
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {DEVICE}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Setup run directory
    TIME_STR = RUN_ID if RUN_ID else datetime.now().strftime("%d%H%M%S")
    LOG_DIR = os.path.join(LOGS_BASE_DIR, TIME_STR)
    MODEL_SAVE_DIR = os.path.join(MODELS_BASE_DIR, TIME_STR)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"Run ID: {TIME_STR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Model save directory: {MODEL_SAVE_DIR}")

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Define controllers
    env = InvePendulum(dt=DT)
    controllers = {"LQR": LQR(-2.91, -3.67, -25.43, -4.94), "SM": SlidingMode(env), "VF": LQR(0, -33.90, -153.30, -32.07)}

    # Train each controller's network
    for controller_name in controllers.keys():
        print(f"\n{'='*60}")
        print(f"Training {controller_name} Controller Network")
        print(f"{'='*60}")

        # Load and preprocess data
        states, targets, df = load_and_preprocess_dataset(controller_name, DATASET_DIR)
        if states is None or len(states) == 0:
            print(f"Skipping {controller_name} due to missing data")
            continue

        # Create normalized datasets
        train_dataset, val_dataset, scalers = create_normalized_datasets(states, targets)

        print(f"  Training set: {len(train_dataset)} samples")
        print(f"  Validation set: {len(val_dataset)} samples")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        # Initialize model, loss, optimizer
        model = NNModel().to(DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

        # Checkpoint paths
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{controller_name.lower()}_checkpoint.pth")
        csv_log_path = os.path.join(LOG_DIR, f"{controller_name.lower()}_training_log.csv")
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{controller_name.lower()}_model.pth")
        scalers_path = os.path.join(MODEL_SAVE_DIR, f"{controller_name.lower()}_scalers.joblib")

        # Save scalers
        joblib.dump(scalers, scalers_path)
        print(f"Saved scalers to {scalers_path}")

        # Training state
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0

        # Resume from checkpoint if available
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        else:
            print("Starting new training session")
            with open(csv_log_path, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_rmse', 'val_rmse', 'learning_rate', 'duration_sec'])

        # Training loop
        for epoch in range(start_epoch, MAX_EPOCHS):
            epoch_start_time = time.time()

            # Train and validate
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, DEVICE)
            val_loss = validate_model(model, val_loader, loss_function, DEVICE)
            current_lr = optimizer.param_groups[0]['lr']

            # Update scheduler
            scheduler.step(val_loss)
            epoch_duration = time.time() - epoch_start_time

            # Calculate RMSE for interpretation
            train_rmse = calculate_rmse(train_loss)
            val_rmse = calculate_rmse(val_loss)

            # Log metrics
            writer.add_scalar(f'{controller_name}/Loss/Train', train_loss, epoch)
            writer.add_scalar(f'{controller_name}/Loss/Validation', val_loss, epoch)
            writer.add_scalar(f'{controller_name}/RMSE/Train', train_rmse, epoch)
            writer.add_scalar(f'{controller_name}/RMSE/Validation', val_rmse, epoch)
            writer.add_scalar(f'{controller_name}/Learning_Rate', current_lr, epoch)

            # Save to CSV
            with open(csv_log_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch, train_loss, val_loss, train_rmse, val_rmse, current_lr, epoch_duration])

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), final_model_path)
                print(f"✓ New best model: Val loss = {val_loss:.6f}, RMSE = {val_rmse:.3f}")
            else:
                patience_counter += 1

            # Print progress
            if epoch % 1 == 0 or epoch < 10:
                print(f"Epoch {epoch:4d} | "
                      f"Train: {train_loss:.6f} ({train_rmse:.3f}) | "
                      f"Val: {val_loss:.6f} ({val_rmse:.3f}) | "
                      f"Best: {best_val_loss:.6f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {epoch_duration:.2f}s")

            # Check stopping conditions
            if train_loss <= EARLY_STOPPING_THRESHOLD:
                print(f"✓ Early stopping: Training loss {train_loss:.6f} <= threshold {EARLY_STOPPING_THRESHOLD}")
                break

            if patience_counter >= PATIENCE_LIMIT:
                print(f"✓ Early stopping: No improvement for {PATIENCE_LIMIT} epochs")
                break

            # Save checkpoint every 25 epochs
            if epoch % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)

        print(f"✓ Finished training {controller_name}. Best validation loss: {best_val_loss:.6f}")
        print(f"Final model saved to: {final_model_path}")

    writer.close()
    print(f"\n✓ Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All models saved to: {MODEL_SAVE_DIR}")
