import pandas as pd
import numpy as np

data = pd.read_csv('Spotify_clean.csv')

np.random.seed(42)

num_rows = data.shape[0]
indexes = np.random.permutation(num_rows)
test_size = int(0.2*num_rows)
test_idx = indexes[:test_size]
train_idx = indexes[test_size:]

train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

train_data.to_csv('Train.csv', index=False)
test_data.to_csv('Test.csv', index=False)