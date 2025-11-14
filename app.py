import gradio as gr
import pandas as pd
import numpy as np
import torch, joblib, pickle
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarPriceModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.2):
        super().__init__()
        layers = []
        sz = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sz) - 1):
            layers.append(torch.nn.Linear(sz[i], sz[i + 1]))
            if i < len(sz) - 2:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def verify_feature_alignment(df, preproc):
    expected_cols = preproc["order"]
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        print(f"⚠️ Alignment issue: missing={missing[:5]} extra={extra[:5]}")
    df = df.reindex(columns=expected_cols, fill_value=0)
    return df

class CarPricePredictor:
    def __init__(self):
        self.model = CarPriceModel(37,  # embeddings + encodings +numeric
                                   [64, 32])
        self.model.load_state_dict(torch.load("best_car_price_model.pt", map_location=device))
        self.model.eval()
        self.model.to(device)
        
        self.encoder = joblib.load("encoder.pkl")
        self.scaler = joblib.load("scaler.pkl")
        self.price_scaler = joblib.load("price_scaler.pkl")
        self.pca_make = joblib.load("pca_make.pkl")
        self.pca_model = joblib.load("pca_model.pkl")
        with open("preproc_params.pkl", "rb") as f:
            self.preproc = pickle.load(f)
        self.transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def predict(self, inputs):
        df = pd.DataFrame([inputs])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        make_col = df["make"].copy()
        model_col = df["model"].copy()
        df["make_emb"] = list(self.pca_make.transform(self.transformer.encode(make_col.tolist())))
        df["model_emb"] = list(self.pca_model.transform(self.transformer.encode(model_col.tolist())))
        df = df.drop(columns=["make", "model"])
        df = pd.concat([
            df.drop(columns=["make_emb","model_emb"]),
            pd.DataFrame(df["make_emb"].tolist(), columns=[f"make_emb_{i}" for i in range(self.pca_make.n_components_)]),
            pd.DataFrame(df["model_emb"].tolist(), columns=[f"model_emb_{i}" for i in range(self.pca_model.n_components_)])
        ], axis=1)
        mean_encode_cols = ["body_type", "exterior_color", "interior_color", "trim"]
        for col in mean_encode_cols:
            mapping = self.preproc["mean_maps"].get(col, {})
            df[col] = df[col].map(mapping).fillna(self.preproc["global_mean"])
        cat_cols = ["transmission","fuel_type","drivetrain","accident_history","seller_type","condition"]
        ohe = self.encoder.transform(df[cat_cols])
        ohe_df = pd.DataFrame(ohe, columns=self.encoder.get_feature_names_out(cat_cols))
        df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)
        df["vehicle_age"] = 2025 - df["year"]
        df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0,1e-6)
        brand_popularity_map = self.preproc.get("brand_popularity_map", {})
        global_pop = self.preproc.get("global_brand_popularity", 0.0)
        df["brand_popularity"] = make_col.map(brand_popularity_map).fillna(global_pop)
        df = verify_feature_alignment(df, self.preproc)
        df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
        x = torch.tensor(df.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            y = self.model(x).cpu().item()
        y = self.price_scaler.inverse_transform([[y]])[0][0]
        return f"🏷️ Estimated Price: ${max(0,y):,.2f}"

predictor = CarPricePredictor()

inputs = [
    gr.Textbox(label="Make"), gr.Textbox(label="Model"),
    gr.Number(label="Year", value=2020), gr.Number(label="Mileage", value=30000),
    gr.Number(label="Engine HP", value=200), gr.Number(label="Owner Count", value=1),
    gr.Dropdown(["Automatic","Manual"], label="Transmission"),
    gr.Dropdown(["Gasoline","Diesel","Electric","Hybrid"], label="Fuel Type"),
    gr.Dropdown(["FWD","RWD","AWD"], label="Drivetrain"),
    gr.Dropdown(["None","Minor","Major"], label="Accident History"),
    gr.Dropdown(["Dealer","Private"], label="Seller Type"),
    gr.Dropdown(["Excellent","Good","Fair"], label="Condition"),
    gr.Textbox(label="Trim"),
    gr.Textbox(label="Exterior Color"), gr.Textbox(label="Interior Color"),
    gr.Textbox(label="Body Type")
]

def predict_car_price(*args):
    keys = ["make","model","year","mileage","engine_hp","owner_count","transmission","fuel_type",
            "drivetrain","accident_history","seller_type","condition","trim",
            "exterior_color", "interior_color", "body_type"]
    inputs = dict(zip(keys,args))
    return predictor.predict(inputs)

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
