import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
import pandas as pd

class AutoFeaturesExtraction:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do automatycznego ekstrahowania cech, YAMNet"

    def load_model(self) -> tf.keras.Model:
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)

        return yamnet_model
    
    def convert_audio_file_to_16kHz(self, file_path: str) -> float:
        pass
