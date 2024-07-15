import numpy as np
import librosa

class MoreFeaturesHandler():
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do ekstrakcji cech z plików audio w postaci wektorowej"
    
    def zero_crossing_rate(self, audio: np.array) -> float:
        return librosa.feature.zero_crossing_rate(audio)