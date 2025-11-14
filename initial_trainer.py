import os, pickle, warnings, joblib
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp

from sentence_transformers import SentenceTransformer

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Model definition

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
    
#Trainer definition

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
        self.start_epoch = 1
        self.best_val = float("inf")
        self.patience_counter = 0

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
        preds_all = np.concatenate(preds_all)
        targ_all = np.concatenate(targ_all)
        return total_loss / len(loader), preds_all, targ_all


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

    def fit(self, train_loader, val_loader, epochs=80, early_stop_patience=12):

        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)

            if epoch % self.val_interval == 0:
                val_loss, _, _ = self.validate(val_loader)
                self.scheduler.step(val_loss)

                print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

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
                        break

        self.model.load_state_dict(torch.load("best_car_price_model.pt"))
        print("✅ Best model restored.")

# ---------------------------
# Data
# ---------------------------
df = pd.read_csv("vehicle_price_prediction.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train={len(train_df)} | Test={len(test_df)}")

# Sentence embeddings for 'make' and 'model'
transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def compute_embeddings(series, name, fit_pca=True, pca_model=None):
    os.makedirs("embedding_cache", exist_ok=True)
    cache = f"embedding_cache/{name}_emb.pt"
    if os.path.exists(cache):
        emb = torch.load(cache)
    else:
        emb = torch.tensor(transformer.encode(series.tolist(), show_progress_bar=True), dtype=torch.float32)
        torch.save(emb, cache)
    if fit_pca:
        pca = PCA(n_components=5)
        reduced = pca.fit_transform(emb.numpy())
        joblib.dump(pca, f"pca_{name}.pkl")
    else:
        reduced = pca_model.transform(emb.numpy())
    return torch.tensor(reduced, dtype=torch.float32)

train_make_emb = compute_embeddings(train_df["make"], "make", fit_pca=True)
train_model_emb = compute_embeddings(train_df["model"], "model", fit_pca=True)

make_pca = joblib.load("pca_make.pkl")
model_pca = joblib.load("pca_model.pkl")

test_make_emb = torch.tensor(make_pca.transform(transformer.encode(test_df["make"].tolist())), dtype=torch.float32)
test_model_emb = torch.tensor(model_pca.transform(transformer.encode(test_df["model"].tolist())), dtype=torch.float32)

def replace_with_embeddings(df, emb, prefix):
    emb_df = pd.DataFrame(emb.numpy(), columns=[f"{prefix}_emb_{i}" for i in range(emb.shape[1])])
    df = df.drop(columns=[prefix]).reset_index(drop=True)
    return pd.concat([df, emb_df], axis=1)

train_df = replace_with_embeddings(train_df, train_make_emb, "make")
train_df = replace_with_embeddings(train_df, train_model_emb, "model")
test_df = replace_with_embeddings(test_df, test_make_emb, "make")
test_df = replace_with_embeddings(test_df, test_model_emb, "model")

# Mean encoding
mean_encode_cols = ["body_type","exterior_color","interior_color","trim"]
global_mean = train_df["price"].mean()
for col in mean_encode_cols:
    mapping = train_df.groupby(col)["price"].mean().to_dict()
    for d in [train_df, test_df]:
        d[col] = d[col].map(mapping).fillna(global_mean)

# One-Hot Encoding
cat_cols = ["transmission","fuel_type","drivetrain","accident_history","seller_type","condition"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(train_df[cat_cols])
joblib.dump(encoder, "encoder.pkl")

def encode_cats(df):
    arr = encoder.transform(df[cat_cols])
    ohe_df = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols))
    return pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), ohe_df], axis=1)

train_df = encode_cats(train_df)
test_df = encode_cats(test_df)

# Standardization
X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]
X_test = test_df.drop(columns=["price"])
y_test = test_df["price"]

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
joblib.dump(scaler, "scaler.pkl")

price_scaler = StandardScaler()
y_train = pd.Series(price_scaler.fit_transform(y_train.values.reshape(-1,1)).flatten())
y_test = pd.Series(price_scaler.transform(y_test.values.reshape(-1,1)).flatten())
joblib.dump(price_scaler, "price_scaler.pkl")

# DataLoaders
def make_loader(X, y, batch=128, shuffle=True):
    ds = TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                       torch.tensor(y.values, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
test_loader = make_loader(X_test, y_test, shuffle=False)

#Create model and trainer instances

model = HuggingFaceRegressor(X_train.shape[1], [64, 32], 1, dropout=0.0)
trainer = CarPriceTrainer(model, lr=0.001, val_interval=1)

#Perform training
trainer.fit(train_loader, test_loader, epochs=80, early_stop_patience=12)

# Evaluate
def evaluate(loader, label):
    _, preds, targets = trainer.validate(loader)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    print(f"\n📊 {label} Set:")
    print(f"   MSE={mse:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

evaluate(test_loader, "Test")

# Save preprocessing

brand_popularity_map = df.groupby("make")["brand_popularity"].mean().to_dict()
global_brand_popularity = df["brand_popularity"].mean()

preproc = {
    "order": list(X_train.columns),   # feature order
    "global_mean": global_mean,        # for unknown categories
    "mean_maps": {},                    # dictionary for each encoded column
    "brand_popularity_map": brand_popularity_map,
    "global_brand_popularity": global_brand_popularity
}

for col in mean_encode_cols:
    preproc["mean_maps"][col] = train_df.groupby(col)["price"].mean().to_dict()

pickle.dump(preproc, open("preproc_params.pkl", "wb"))

print("✅ Training complete and preprocessing saved.")