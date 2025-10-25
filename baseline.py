import pandas as pd
import numpy as np

df = pd.read_csv('vehicle_price_prediction.csv')

y = df['price']
price_baseline = np.std(y)
print(f'Price Baseline (Standard Deviation): {price_baseline}')