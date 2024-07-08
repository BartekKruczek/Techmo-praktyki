import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

class DataLoaderHandler(Dataset):
    def __init__(self, dataframe: pd.DataFrame, device: torch.device) -> None:
        self.dataframe = dataframe
        self.device = device

    def __repr__(self) -> str:
        return "Klasa do ładowania danych, podejście 2.0"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def dataframe_cleaner(self) -> pd.DataFrame:
        df = self.dataframe

        # drop rows with missing values
        df = df.dropna()

        # iterate in healthy_status column, if pathological 1, if healthy 0
        df['healthy_status'] = df['healthy_status'].apply(lambda x: 1 if x == 'pathological' else 0)

        return df
    
    def __getitem__(self, idx: int):
        dataframe = self.dataframe_cleaner()

        audio_path = dataframe.iloc[idx]['file_path']
        label = dataframe.iloc[idx]['healthy_status']

        # load audio file, extract MFCC features
        try:
            y, sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # convert to tensor
        mfcc = torch.tensor(mfcc, dtype=torch.float32).to(self.device)
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return mfcc, label
