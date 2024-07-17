import numpy as np
import librosa
import torch

class MoreFeaturesHandler():
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do ekstrakcji cech z plików audio w postaci wektorowej"
    
    def zero_crossing_rate(self, audio: np.array) -> float:
        return librosa.feature.zero_crossing_rate(audio)
    
    def spectral_centroid(self, audio: np.array, sr: int) -> float:
        return librosa.feature.spectral_centroid(audio, sr = sr)
    
    def delta_delta_mfcc(self, audio: np.array, sr: int) -> float:
        return librosa.feature.delta(librosa.feature.mfcc(audio, sr = sr))
    
    def combined_features(self, audio: np.array, sr: int) -> np.array:
        zcr = self.zero_crossing_rate(audio)
        sc = self.spectral_centroid(audio, sr)
        ddm = self.delta_delta_mfcc(audio, sr)
        
        # torch vstack
        return np.vstack((zcr, sc, ddm), dtype = torch.float32)