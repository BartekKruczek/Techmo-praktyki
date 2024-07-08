import pandas as pd
import librosa

from torch.utils.data import Dataset

class DataLoaderHandler(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.dataframe = dataframe

    def __repr__(self) -> str:
        return "Klasa do ładowania danych, podejście 2.0"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int):
        audio_path = self.dataframe.iloc[idx]['file_path']
        label = self.dataframe.iloc[idx]['healthy_status']

        # load audio file, extract MFCC features
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)

        return mfcc, label