import gradio as gr
import pandas as pd
import numpy as np
import torch, joblib, pickle
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# 🔧 Model architecture used during training + inference
# -------------------------------------------------------
class CarPriceModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.2):
        super().__init__()
        layers = []
        # Build MLP: input → hidden layers → output
        sz = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sz) - 1):
            layers.append(torch.nn.Linear(sz[i], sz[i + 1]))
            if i < len(sz) - 2:  # hidden layers only
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------
# 🧩 Ensure inference features match training order
# -------------------------------------------------------
def verify_feature_alignment(df, preproc):
    expected_cols = preproc["order"]  # list of features used during training

    # Warn if anything is missing or extra
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        print(f"⚠️ Alignment issue: missing={missing[:5]} extra={extra[:5]}")

    # Reorder and fill missing columns with zeros
    df = df.reindex(columns=expected_cols, fill_value=0)
    return df

# -------------------------------------------------------
# 🚗 Main Predictor Class
# -------------------------------------------------------
class CarPricePredictor:
    def __init__(self):
        # Load trained PyTorch model
        self.model = CarPriceModel(37, [64, 32])
        self.model.load_state_dict(torch.load("best_car_price_model.pt", map_location=device))
        self.model.eval()
        self.model.to(device)

        # Load preprocessing artifacts
        self.encoder = joblib.load("encoder.pkl")             # One-hot encoder
        self.scaler = joblib.load("scaler.pkl")               # Feature scaler
        self.price_scaler = joblib.load("price_scaler.pkl")   # Target scaler
        self.pca_make = joblib.load("pca_make.pkl")           # PCA for make embeddings
        self.pca_model = joblib.load("pca_model.pkl")         # PCA for model embeddings

        # Load custom preprocessing params (mean encoding, feature order, etc.)
        with open("preproc_params.pkl", "rb") as f:
            self.preproc = pickle.load(f)

        # Sentence transformer for generating embeddings
        self.transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # -------------------------------------------------------
    # 🔮 Predict single input
    # -------------------------------------------------------
    def predict(self, inputs):
        # Convert dictionary → DataFrame
        df = pd.DataFrame([inputs])

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Save original make/model for later use
        make_col = df["make"].copy()
        model_col = df["model"].copy()

        # 1️⃣ Encode make/model → sentence embeddings → PCA
        df["make_emb"] = list(self.pca_make.transform(self.transformer.encode(make_col.tolist())))
        df["model_emb"] = list(self.pca_model.transform(self.transformer.encode(model_col.tolist())))

        # Drop raw text columns
        df = df.drop(columns=["make", "model"])

        # Expand embedding vectors into individual numeric columns
        df = pd.concat([
            df.drop(columns=["make_emb","model_emb"]),
            pd.DataFrame(df["make_emb"].tolist(),  columns=[f"make_emb_{i}"  for i in range(self.pca_make.n_components_)]),
            pd.DataFrame(df["model_emb"].tolist(), columns=[f"model_emb_{i}" for i in range(self.pca_model.n_components_)])
        ], axis=1)

        # 2️⃣ Mean encoding for select features
        mean_encode_cols = ["body_type", "exterior_color", "interior_color", "trim"]
        for col in mean_encode_cols:
            mapping = self.preproc["mean_maps"].get(col, {})
            df[col] = df[col].map(mapping).fillna(self.preproc["global_mean"])

        # 3️⃣ One-hot encoding categorical fields
        cat_cols = ["transmission","fuel_type","drivetrain","accident_history","seller_type","condition"]
        ohe = self.encoder.transform(df[cat_cols])
        ohe_df = pd.DataFrame(ohe, columns=self.encoder.get_feature_names_out(cat_cols))
        df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)

        # 4️⃣ Add engineered features
        df["vehicle_age"] = 2025 - df["year"]
        df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0,1e-6)

        # Brand popularity (encoded from training stats)
        brand_popularity_map = self.preproc.get("brand_popularity_map", {})
        global_pop = self.preproc.get("global_brand_popularity", 0.0)
        df["brand_popularity"] = make_col.map(brand_popularity_map).fillna(global_pop)

        # Ensure features match training order
        df = verify_feature_alignment(df, self.preproc)

        # 5️⃣ Scale numeric features
        df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        # Convert to tensor
        x = torch.tensor(df.values, dtype=torch.float32).to(device)

        # 6️⃣ Run model
        with torch.no_grad():
            y = self.model(x).cpu().item()

        # 7️⃣ Convert scaled prediction back to actual price
        y = self.price_scaler.inverse_transform([[y]])[0][0]

        return f"🏷️ Estimated Price: ${max(0,y):,.2f}"

# Initialize predictor
predictor = CarPricePredictor()

# -------------------------------------------------------
# 🎨 Gradio Interface
# -------------------------------------------------------
inputs = [
    gr.Textbox(label="Make"),
    gr.Textbox(label="Model"),
    gr.Number(label="Year", value=2020),
    gr.Number(label="Mileage", value=30000),
    gr.Number(label="Engine HP", value=200),
    gr.Number(label="Owner Count", value=1),
    gr.Dropdown(["Automatic","Manual"], label="Transmission"),
    gr.Dropdown(["Gasoline","Diesel","Electric","Hybrid"], label="Fuel Type"),
    gr.Dropdown(["FWD","RWD","AWD"], label="Drivetrain"),
    gr.Dropdown(["None","Minor","Major"], label="Accident History"),
    gr.Dropdown(["Dealer","Private"], label="Seller Type"),
    gr.Dropdown(["Excellent","Good","Fair"], label="Condition"),
    gr.Textbox(label="Trim"),
    gr.Textbox(label="Exterior Color"),
    gr.Textbox(label="Interior Color"),
    gr.Textbox(label="Body Type")
]

# Wrap predictor for Gradio
def predict_car_price(*args):
    keys = ["make","model","year","mileage","engine_hp","owner_count","transmission","fuel_type",
            "drivetrain","accident_history","seller_type","condition","trim",
            "exterior_color", "interior_color", "body_type"]
    inputs = dict(zip(keys,args))
    return predictor.predict(inputs)

# Gradio app
iface = gr.Interface(
    fn=predict_car_price,
    inputs=inputs,
    outputs=gr.Label(label="Predicted Price"),
    title="🚗 Car Price Predictor",
    description="Predicts car price using a trained deep learning model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    print([input.label for input in inputs])
    iface.launch(server_name="0.0.0.0", server_port=7860)
