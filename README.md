Project Overview

This project is a deep learning–based system that predicts the market price of cars using a combination of numeric, categorical, and text-derived features. The system uses Sentence Transformer embeddings, PCA reduction, One-Hot Encoding, Mean Encoding, Scaling, and a fully connected neural network trained using PyTorch.

## Project Overview
This project predicts used car prices using numeric, categorical, and text-derived features. It combines:
- Sentence Transformer embeddings
- PCA reduction
- One-Hot Encoding
- Mean Encoding
- Scaling
- Fully connected PyTorch neural network

## Project Goals
- Learn representations from textual fields (make/model)
- Incorporate structured vehicle characteristics
- Apply consistent preprocessing
- Use GPU-accelerated deep learning
- Provide a user-friendly Gradio interface

The neural network's architecture looks roughly like this:

Input → Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Linear(1)

## Characteristcs:

	- Mixed-precision training via torch.cuda.amp

	- The Adam optimizer

	- The ReduceLROnPlateau scheduler


## Web app features:

	- A clean Gradio interface with input fields

	- Textboxes and dropdowns for user-entered car details

	- Real-time preprocessing identical to training

	- Calls to the trained PyTorch model

	- Output displayed as a formatted price prediction


## Inference Pipeline:

	1. Embedding generation → PCA reduction

	2. Mean encoding

	3. One-hot encoding

	4. Feature alignment

	5. Scaling

	6. Model inference

	7. Inverse-scaling of predicted price

## Data Requirements

- CSV file: [vehicle_price_prediction.csv](https://www.kaggle.com/datasets/metawave/vehicle-price-prediction)
- Python version: 3.11.0

## Library Requirements
### Deep Learning Stack
- `torch==2.7.0`  
- `torchvision==0.22.0`  
- `torchaudio==2.7.0`  
Optional CUDA 12.8 support for GPU acceleration.

### NLP / Embeddings
- `sentence-transformers==2.7.0`  
- `transformers==4.44.2`  

### Web Interface
- `gradio==4.44.1`  

### Scientific / ML Libraries
- `pandas==2.2.3`  
- `numpy==1.26.4`  
- `scikit-learn==1.5.2`  
- `joblib==1.4.2`  

## Training Artifacts
Running the trainer produces:
- `best_car_price_model.pt`  
- `training_checkpoint.pt`  
- `encoder.pkl`  
- `scaler.pkl`  
- `price_scaler.pkl`  
- `pca_make.pkl`  
- `pca_model.pkl`  
- `preproc_params.pkl`  
- `embedding_cache/` containing `make_emb.pt` and `model_emb.pt`

In order to perform training and run the app successfully, it is recommended to keep all the files within the same folder.
