import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchaudio.transforms import Resample

class PytorchModelHandler(Dataset):
    def __init__(self, dataframe: pd.DataFrame, augmentation: bool, classification: bool) -> None:
        self.classification = classification
        self.dataframe = dataframe
        self.dataframe = self.dataframe_cleaner()
        self.augmentation = augmentation

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Pytorch"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> tuple[dict, int]:
        """
        Zwraca słownik cech oraz przypisaną etykietę.
        """
        list_of_audio_paths = self.get_audio_paths()
        list_of_audio_labels = self.get_labels()
        demographic = self.dataframe.iloc[idx]['healthy_status_gender']

        if len(list_of_audio_paths) == len(list_of_audio_labels):
            audio_path = list_of_audio_paths[idx]
            label = list_of_audio_labels[idx]

            audio, sr = self.get_audio(audio_path)

            if self.augmentation:
                audio = self.apply_random_augmentation(audio)

            if audio is not None and sr is not None:
                features = self.extract_features(audio, sr)
                return features, label, demographic
            else:
                return None, None, None

    def dataframe_cleaner(self) -> pd.DataFrame:
        df = self.dataframe

        # drop rows with missing values
        df = df.dropna()

        if self.classification != "multi":
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
    
    def change_sr(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        return Resample(orig_freq=sr, new_freq=16000)(audio)
        
    def extract_features(self, audio: torch.Tensor, sr: int) -> dict:
        audio_np = audio.squeeze().numpy()
        zcr = librosa.feature.zero_crossing_rate(y=audio_np)
        sc = librosa.feature.spectral_centroid(y=audio_np, sr=sr)
        ddm = librosa.feature.delta(librosa.feature.mfcc(y=audio_np, sr=sr))

        # stack features
        features = np.vstack((zcr, sc, ddm)).T

        return torch.tensor(features, dtype=torch.float32)
    
    def get_features(self) -> torch.Tensor:
        list_of_audio_paths = self.get_audio_paths()

        for elem in list_of_audio_paths:
            try:
                audio, sr = self.get_audio(elem)
                features = self.extract_features(audio, sr)
                return features
            except Exception as e:
                print(f"Error: {e}")
                continue
        
    def apply_random_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        augmentations = [self.add_noise, self.add_shift, self.add_stretch]
        augmentation = augmentations[torch.randint(0, len(augmentations), (1,)).item()]

        return augmentation(audio)

    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        noise_factor = 0.005

        noise = torch.randn(audio.size())
        noisy_audio = audio + noise_factor * noise

        return noisy_audio

    def add_shift(self, audio: torch.Tensor) -> torch.Tensor:
        shift = int(audio.size(1) * 0.1)

        return torch.roll(audio, shifts=shift, dims=1)
    
    def add_stretch(self, audio: torch.Tensor) -> torch.Tensor:
        stretch_factor = 1.2
        stretch_length = int(audio.size(1) * stretch_factor)
        stretched_audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=(stretch_length,), mode='nearest').squeeze(0)

        return stretched_audio
