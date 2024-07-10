import pandas as pd
import torch
import torchaudio

from torch.utils.data import Dataset
from torch.nn import functional as F

class DataLoaderHandler(Dataset):
    def __init__(self, dataframe: pd.DataFrame, device: torch.device, augmentation: bool) -> None:
        self.dataframe = dataframe
        self.device = device
        self.augmentation = augmentation

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

        # load audio file, extract mel spectrogram
        try:
            y, sr = torchaudio.load(audio_path, normalize = True)

            if self.augmentation:
                y = self.apply_random_augmentation(y)

            transform = torchaudio.transforms.MelSpectrogram(sample_rate = sr)
            mel_spectrogram = transform(y)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # convert to tensor, send to device
        mel_spectrogram = mel_spectrogram.to(self.device)
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return mel_spectrogram, label
    
    def apply_random_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        # randomly select augmentation
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
        stretched_audio = F.interpolate(audio.unsqueeze(0), size=(stretch_length,), mode='nearest').squeeze(0)

        return stretched_audio