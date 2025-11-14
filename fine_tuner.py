import os, pickle, joblib, torch, warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch.cuda import amp

# -------------------------------
# Reproducibility setup
# -------------------------------
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True   # Ensures deterministic conv operations
torch.backends.cudnn.benchmark = False      # Disable autotuner for reproducibility

# -------------------------------
# Device selection
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Model Definition
# ============================================================

class HuggingFaceRegressor(nn.Module):
    """Simple feed-forward neural regressor used after embeddings + preprocessing."""
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.0):
        super().__init__()

        layers = []
        sz = [input_size] + hidden_sizes + [output_size]  # layer sizes including output

        # Build fully-connected network dynamically
        for i in range(len(sz) - 1):
            layers.append(nn.Linear(sz[i], sz[i + 1]))  # linear layer
            if i < len(sz) - 2:                        # add activations for hidden layers only
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # forward pass


# ============================================================
# Training + Checkpointing Class
# ============================================================

class CarPriceTrainer:
    """Handles training, validation, checkpointing, and resume logic."""
    def __init__(self, model, lr=0.001, val_interval=1, checkpoint_path="training_checkpoint.pt"):
        self.model = model.to(device)

        # Enable mixed precision on GPU for speed
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        self.criterion = nn.MSELoss()  # regression objective
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-5)

        # Reduce learning rate when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.val_interval = val_interval
        self.checkpoint_path = checkpoint_path
        self.patience_counter = 0

        # Try loading previous training state
        self.load_checkpoint()

    # ------------------------------
    # Save checkpoint to disk
    # ------------------------------
    def save_checkpoint(self, epoch, best_val):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val": best_val,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch} (val_loss={best_val:.6f})")

    # ------------------------------
    # Load checkpoint if available
    # ------------------------------
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)

            # Restore model, optimizer, scheduler, AMP
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.scaler.load_state_dict(checkpoint["scaler_state"])

            self.best_val = checkpoint.get("best_val", float("inf"))
            self.start_epoch = checkpoint.get("epoch", 1) + 1  # start after last saved epoch

            print(f"🔄 Checkpoint loaded (resuming from epoch {self.start_epoch})")
        else:
            print("⚠️ No existing checkpoint found, starting from scratch.")
            self.best_val = float("inf")
            self.start_epoch = 1

    # ------------------------------
    # One training epoch
    # ------------------------------
    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with amp.autocast():
                preds = self.model(xb)
                loss = self.criterion(preds, yb)

            # Backprop with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------
    # Validation loop
    # ------------------------------
    def validate(self, loader):
        self.model.eval()
        total_loss, preds_all, targ_all = 0.0, [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                with amp.autocast():
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)

                total_loss += loss.item()
                preds_all.append(preds.cpu().numpy())
                targ_all.append(yb.cpu().numpy())

        preds_all = np.concatenate(preds_all)
        targ_all = np.concatenate(targ_all)

        return total_loss / len(loader), preds_all, targ_all

    # ------------------------------
    # Full training loop (supports resume)
    # ------------------------------
    def fit(self, train_loader, val_loader, epochs=80, early_stop_patience=12, resume=True):
        if resume:
            self.load_checkpoint()

        # Continue from the checkpoint epoch
        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            # Train for 1 epoch
            train_loss = self.train_one_epoch(train_loader)

            # Validate at intervals
            if epoch % self.val_interval == 0:
                val_loss, _, _ = self.validate(val_loader)
                self.scheduler.step(val_loss)  # update LR based on validation loss

                print(f"Epoch {epoch:3d}/{self.start_epoch + epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f}")

                # Check for improvement
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.patience_counter = 0

                    torch.save(self.model.state_dict(), "best_car_price_model.pt")
                    self.save_checkpoint(epoch, self.best_val)

                else:
                    self.patience_counter += 1
                    self.save_checkpoint(epoch, self.best_val)

                    if self.patience_counter >= early_stop_patience:
                        print("⏸️ Early stopping triggered.")
                        self.patience_counter = 0
                        break

        # Restore best model
        self.model.load_state_dict(torch.load("best_car_price_model.pt"))
        print("✅ Best model restored.")


# ============================================================
# Load previously saved preprocessing pipeline
# ============================================================

encoder = joblib.load("encoder.pkl")              # OHE encoder
scaler = joblib.load("scaler.pkl")                # Feature scaler
price_scaler = joblib.load("price_scaler.pkl")    # Target scaler
make_pca = joblib.load("pca_make.pkl")            # PCA for "make" embeddings
model_pca = joblib.load("pca_model.pkl")          # PCA for "model" embeddings
preproc = pickle.load(open("preproc_params.pkl", "rb"))  # Additional preprocessing state

mean_encode_cols = list(preproc["mean_maps"].keys())
brand_popularity_map = preproc["brand_popularity_map"]
global_mean = preproc["global_mean"]
global_brand_popularity = preproc["global_brand_popularity"]
feature_order = preproc["order"]  # Needed so features align exactly as during training

transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ============================================================
# Load fresh dataset (train/test split)
# ============================================================

df = pd.read_csv("vehicle_price_prediction.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train={len(train_df)} | Test={len(test_df)}")

# ============================================================
# Apply SAME preprocessing as original training
# ============================================================

def load_embeddings(series, name, pca):
    """Load cached embeddings and apply saved PCA reduction."""
    cache_path = f"embedding_cache/{name}_emb.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached embedding not found: {cache_path}. Run initial training first.")
    emb = torch.load(cache_path)
    reduced = pca.transform(emb.numpy())
    return torch.tensor(reduced, dtype=torch.float32)

def replace_with_embeddings(df, emb, prefix):
    """Replace text column with its reduced embedding vector."""
    emb_df = pd.DataFrame(emb.numpy(), columns=[f"{prefix}_emb_{i}" for i in range(emb.shape[1])])
    df = df.drop(columns=[prefix]).reset_index(drop=True)
    return pd.concat([df, emb_df], axis=1)

# Load + apply saved PCA on sentence embeddings
train_make_emb = load_embeddings(train_df["make"], "make", make_pca)
train_model_emb = load_embeddings(train_df["model"], "model", model_pca)

# Test split: compute new embeddings but reduce using same PCA
test_make_emb = torch.tensor(make_pca.transform(transformer.encode(test_df["make"].tolist())), dtype=torch.float32)
test_model_emb = torch.tensor(model_pca.transform(transformer.encode(test_df["model"].tolist())), dtype=torch.float32)

# Replace text columns with embeddings
train_df = replace_with_embeddings(train_df, train_make_emb, "make")
train_df = replace_with_embeddings(train_df, train_model_emb, "model")
test_df = replace_with_embeddings(test_df, test_make_emb, "make")
test_df = replace_with_embeddings(test_df, test_model_emb, "model")

# Mean encoding using saved mappings
for col in mean_encode_cols:
    mapping = preproc["mean_maps"].get(col, {})
    for d in [train_df, test_df]:
        d[col] = d[col].map(mapping).fillna(global_mean)

# One-hot encoding
cat_cols = ["transmission","fuel_type","drivetrain","accident_history","seller_type","condition"]

def encode_cats(df):
    """Apply saved one-hot encoder."""
    arr = encoder.transform(df[cat_cols])
    ohe_df = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols))
    return pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), ohe_df], axis=1)

train_df = encode_cats(train_df)
test_df = encode_cats(test_df)

# Ensure identical feature order (important!)
for col in feature_order:
    train_df.setdefault(col, 0.0)
    test_df.setdefault(col, 0.0)

train_df = train_df[feature_order + ["price"]]
test_df = test_df[feature_order + ["price"]]

# Scale features and targets
X_train = pd.DataFrame(scaler.transform(train_df.drop(columns=["price"])), columns=feature_order)
X_test = pd.DataFrame(scaler.transform(test_df.drop(columns=["price"])), columns=feature_order)

y_train = pd.Series(price_scaler.transform(train_df["price"].values.reshape(-1,1)).flatten())
y_test = pd.Series(price_scaler.transform(test_df["price"].values.reshape(-1,1)).flatten())

# ============================================================
# Create PyTorch dataloaders
# ============================================================

def make_loader(X, y, batch=128, shuffle=True):
    """Convert numpy/pandas data to PyTorch DataLoader."""
    ds = TensorDataset(
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
test_loader = make_loader(X_test, y_test, shuffle=False)

# ============================================================
# Load model for continued training
# ============================================================

input_size = len(feature_order)
model = HuggingFaceRegressor(input_size, [64, 32], 1, dropout=0.0)
model.load_state_dict(torch.load("best_car_price_model.pt", map_location=device))

# Lower LR for fine-tuning
trainer = CarPriceTrainer(model, lr=0.0005, val_interval=1)

# ============================================================
# Continue training
# ============================================================

print("\n🚀 Continuing training on the same dataset...")
trainer.fit(train_loader, test_loader, epochs=20, early_stop_patience=8)

# Save updated model weights
torch.save(model.state_dict(), "best_car_price_model.pt")
print("✅ Continued training complete. Model saved as 'best_car_price_model.pt'.")

# ============================================================
# Evaluate after continued training
# ============================================================

def evaluate(loader, label):
    """Compute MSE, RMSE, and R² score."""
    _, preds, targets = trainer.validate(loader)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    print(f"\n📊 {label} Set:")
    print(f"   MSE={mse:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

evaluate(test_loader, "Validation/Test")
