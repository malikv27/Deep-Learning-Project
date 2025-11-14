import os, pickle, joblib, torch, warnings
warnings.filterwarnings("ignore")

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
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HuggingFaceRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.0):
        super().__init__()
        layers = []
        sz = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sz) - 1):
            layers.append(nn.Linear(sz[i], sz[i + 1]))
            if i < len(sz) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class CarPriceTrainer:
    def __init__(self, model, lr=0.001, val_interval=1, checkpoint_path="training_checkpoint.pt"):
        self.model = model.to(device)
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.val_interval = val_interval
        self.checkpoint_path = checkpoint_path
        self.patience_counter = 0
        self.load_checkpoint()

    # --------------------
    # ✅ Save checkpoint
    # --------------------
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

    # --------------------
    # ✅ Load checkpoint
    # --------------------
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.scaler.load_state_dict(checkpoint["scaler_state"])
            self.best_val = checkpoint.get("best_val", float("inf"))
            self.start_epoch = checkpoint.get("epoch", 1) + 1
            print(f"🔄 Checkpoint loaded (resuming from epoch {self.start_epoch})")
        else:
            print("⚠️ No existing checkpoint found, starting from scratch.")

    # --------------------
    # Training logic
    # --------------------
    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            self.optimizer.zero_grad()
            with amp.autocast():
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(loader)

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
        import numpy as np
        preds_all = np.concatenate(preds_all)
        targ_all = np.concatenate(targ_all)
        return total_loss / len(loader), preds_all, targ_all

    # --------------------
    # ✅ Fit with checkpoint resume
    # --------------------
    def fit(self, train_loader, val_loader, epochs=80, early_stop_patience=12, resume=True):
        if resume:
            self.load_checkpoint()  # try to resume previous run

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            train_loss = self.train_one_epoch(train_loader)

            if epoch % self.val_interval == 0:
                val_loss, _, _ = self.validate(val_loader)
                self.scheduler.step(val_loss)

                print(f"Epoch {epoch:3d}/{self.start_epoch + epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f}")

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

        self.model.load_state_dict(torch.load("best_car_price_model.pt"))
        print("✅ Best model restored.")

# ---------------------------------------------------------
# 1️⃣ Load preprocessing artifacts
# ---------------------------------------------------------
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
price_scaler = joblib.load("price_scaler.pkl")
make_pca = joblib.load("pca_make.pkl")
model_pca = joblib.load("pca_model.pkl")
preproc = pickle.load(open("preproc_params.pkl", "rb"))

mean_encode_cols = list(preproc["mean_maps"].keys())
brand_popularity_map = preproc["brand_popularity_map"]
global_mean = preproc["global_mean"]
global_brand_popularity = preproc["global_brand_popularity"]
feature_order = preproc["order"]

transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ---------------------------------------------------------
# 2️⃣ Load data
# ---------------------------------------------------------
df = pd.read_csv("vehicle_price_prediction.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train={len(train_df)} | Test={len(test_df)}")

# ---------------------------------------------------------
# 3️⃣ Apply the SAME preprocessing
# ---------------------------------------------------------

def load_embeddings(series, name, pca):
    cache_path = f"embedding_cache/{name}_emb.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached embedding not found: {cache_path}. "
                                f"Run initial training first.")
    emb = torch.load(cache_path)
    reduced = pca.transform(emb.numpy())
    return torch.tensor(reduced, dtype=torch.float32)

def replace_with_embeddings(df, emb, prefix):
    emb_df = pd.DataFrame(emb.numpy(), columns=[f"{prefix}_emb_{i}" for i in range(emb.shape[1])])
    df = df.drop(columns=[prefix]).reset_index(drop=True)
    return pd.concat([df, emb_df], axis=1)

# Sentence embeddings (using existing PCA)
train_make_emb = load_embeddings(train_df["make"], "make", make_pca)
train_model_emb = load_embeddings(train_df["model"], "model", model_pca)
test_make_emb = torch.tensor(
    make_pca.transform(transformer.encode(test_df["make"].tolist())), dtype=torch.float32
)
test_model_emb = torch.tensor(
    model_pca.transform(transformer.encode(test_df["model"].tolist())), dtype=torch.float32
)


train_df = replace_with_embeddings(train_df, train_make_emb, "make")
train_df = replace_with_embeddings(train_df, train_model_emb, "model")
test_df = replace_with_embeddings(test_df, test_make_emb, "make")
test_df = replace_with_embeddings(test_df, test_model_emb, "model")

# Mean encoding
for col in mean_encode_cols:
    mapping = preproc["mean_maps"].get(col, {})
    for d in [train_df, test_df]:
        d[col] = d[col].map(mapping).fillna(global_mean)

# One-hot encoding
cat_cols = ["transmission","fuel_type","drivetrain","accident_history","seller_type","condition"]

def encode_cats(df):
    arr = encoder.transform(df[cat_cols])
    ohe_df = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols))
    df = pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), ohe_df], axis=1)
    return df

train_df = encode_cats(train_df)
test_df = encode_cats(test_df)

# Ensure same feature columns
for col in feature_order:
    if col not in train_df.columns:
        train_df[col] = 0.0
    if col not in test_df.columns:
        test_df[col] = 0.0

train_df = train_df[feature_order + ["price"]]
test_df = test_df[feature_order + ["price"]]

# Scale features
X_train = pd.DataFrame(scaler.transform(train_df.drop(columns=["price"])), columns=feature_order)
X_test = pd.DataFrame(scaler.transform(test_df.drop(columns=["price"])), columns=feature_order)
y_train = pd.Series(price_scaler.transform(train_df["price"].values.reshape(-1,1)).flatten())
y_test = pd.Series(price_scaler.transform(test_df["price"].values.reshape(-1,1)).flatten())

# ---------------------------------------------------------
# 4️⃣ Create dataloaders
# ---------------------------------------------------------
def make_loader(X, y, batch=128, shuffle=True):
    ds = TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                       torch.tensor(y.values, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
test_loader = make_loader(X_test, y_test, shuffle=False)

# ---------------------------------------------------------
# 5️⃣ Load model and trainer
# ---------------------------------------------------------
input_size = len(feature_order)
model = HuggingFaceRegressor(input_size, [64, 32], 1, dropout=0.0)
model.load_state_dict(torch.load("best_car_price_model.pt", map_location=device))

trainer = CarPriceTrainer(model, lr=0.0005, val_interval=1)  # lower LR for fine-tuning

# ---------------------------------------------------------
# 6️⃣ Continue training
# ---------------------------------------------------------
print("\n🚀 Continuing training on the same dataset...")
trainer.fit(train_loader, test_loader, epochs=20, early_stop_patience=8)

torch.save(model.state_dict(), "best_car_price_model.pt")
print("✅ Continued training complete. Model saved as 'best_car_price_model.pt'.")

# ---------------------------------------------------------
# 7️⃣ Evaluate
# ---------------------------------------------------------
def evaluate(loader, label):
    _, preds, targets = trainer.validate(loader)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    print(f"\n📊 {label} Set:")
    print(f"   MSE={mse:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

evaluate(test_loader, "Validation/Test")