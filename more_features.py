import numpy as np
import librosa
import torch
import pandas as pd
import torchaudio

from torch.utils.data import Dataset

class MoreFeaturesHandler(Dataset):
    def __init__(self, dataframe: pd.DataFrame, classification: str) -> None:
        self.dataframe = dataframe
        self.classification = classification

    def __repr__(self) -> str:
        return "Klasa do ekstrakcji cech z plików audio w postaci wektorowej"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Zwraca parę w postaci tensora cech w postaci wektorowej oraz przypisaną etykietę.
        """
        list_of_audio_paths = self.get_audio_paths()
        list_of_audio_labels = self.get_labels()

        if len(list_of_audio_paths) == len(list_of_audio_labels):
            audio_path = list_of_audio_paths[idx]
            label = list_of_audio_labels[idx]

            audio, sr = self.get_audio(audio_path)

            print(f"Audio: {audio}, SR: {sr}")

    def dataframe_cleaner(self) -> pd.DataFrame:
        df = self.dataframe

        # drop rows with missing values
        df = df.dropna()

        if self.classification != "multi":
            # iterate in healthy_status column, if pathological 1, if healthy 0
            df['healthy_status'] = df['healthy_status'].apply(lambda x: 1 if x == 'pathological' else 0)
        else:
            class_mapping = {
                'healthy': 0,
                'SLI': 1,
                'Dysarthria': 2,
                'Cyste': 3,
                'Laryngitis': 4,
                'Phonasthenie': 5
            }
            df['healthy_status'] = df['healthy_status'].map(class_mapping)

        return df
    
    def get_audio_paths(self) -> list[str]:
        return self.dataframe['file_path'].values.tolist()
    
    def get_labels(self) -> list[int]:
        return self.dataframe['healthy_status'].values.tolist()
    
    def get_audio(self, file_path) -> tuple[torch.Tensor, int]:
        try:
            y, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return None, None
        return y, sr
    
    def zero_crossing_rate(self, audio: np.array) -> float:
        return librosa.feature.zero_crossing_rate(audio)
    
    def spectral_centroid(self, audio: np.array, sr: int) -> float:
        return librosa.feature.spectral_centroid(audio, sr = sr)
    
    def delta_delta_mfcc(self, audio: np.array, sr: int) -> float:
        return librosa.feature.delta(librosa.feature.mfcc(audio, sr = sr))
    
    def combined_features(self, audio: np.array, sr: int) -> torch.Tensor:
        zcr = self.zero_crossing_rate(audio)
        sc = self.spectral_centroid(audio, sr)
        ddm = self.delta_delta_mfcc(audio, sr)
        
        features = np.vstack((zcr, sc, ddm)).T
        return torch.tensor(features, dtype = torch.float32)