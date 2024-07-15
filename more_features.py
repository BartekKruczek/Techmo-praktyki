import numpy as np
import librosa

class MoreFeaturesHandler():
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do ekstrakcji cech z plików audio w postaci wektorowej"
    
    def zero_crossing_rate(self, audio: np.array) -> float:
        return librosa.feature.zero_crossing_rate(audio)
    
    def spectral_centroid(self, audio: np.array, sr: int) -> float:
        return librosa.feature.spectral_centroid(audio, sr = sr)