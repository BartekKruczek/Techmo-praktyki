import librosa
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

audio_path = 'Database/SVD/Cyste/Pathological/F/Sentence/1340-phrase.wav'
y, sr = librosa.load(audio_path, sr=None)

# mfcc
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_transposed = mfcc.T
print(f"MFCC shape: {mfcc.shape}")
print(f"MFCC transposed shape: {mfcc_transposed.shape}")

dataframe = pd.DataFrame()
dataframe['MFCC'] = [mfcc]
dataframe['MFCC_transposed'] = [mfcc_transposed]
dataframe.head()

mfcc_from_dataframe = dataframe['MFCC']
mfcc_transposed_from_dataframe = dataframe['MFCC_transposed']
print(f"MFCC from dataframe: {mfcc_from_dataframe.shape}")
print(f"MFCC transposed from dataframe: {mfcc_transposed_from_dataframe.shape}")

print(f"MFCC from dataframe type: {type(mfcc_from_dataframe)}")
print(f"MFCC transposed from dataframe type: {type(mfcc_transposed_from_dataframe)}")

# convert pandas Series to numpy array
mfcc_from_dataframe = np.asarray(mfcc_from_dataframe)
print(f"MFCC from dataframe type: {type(mfcc_from_dataframe)}")
print(mfcc_from_dataframe[0].shape)

# flatten the array
mfcc_flattened = mfcc_from_dataframe[0].flatten().astype(np.float64)
print(mfcc_flattened)
print(len(mfcc_flattened))

# convert flattened array to tensor
mfcc_tensor = torch.tensor(mfcc_flattened, dtype=torch.float64)
print(mfcc_tensor)
print(len(mfcc_tensor))

# generate random MFCC values in dataframe['MFCC'], to be 10 rows
dataframe['MFCC'] = [np.random.rand(10, 13) for _ in range(dataframe.shape[0])]

# generate random 0, 1 labels in dataframe['label']
dataframe['label'] = np.random.randint(0, 2, dataframe.shape[0])

# split data
X = dataframe['MFCC']
y = dataframe['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)