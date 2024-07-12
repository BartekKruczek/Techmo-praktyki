import tensorflow as tf
import pandas as pd
import torchaudio
import torch
import torchaudio.transforms as transforms
import numpy as np

class AutoFeaturesExtraction:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do automatycznego ekstrahowania cech, YAMNet"

    def load_model(self) -> tf.keras.Model:
        yamnet_model = tf.keras.layers.TFSMLayer('yamnet-model', call_endpoint='serving_default')

        return yamnet_model
    
    def get_audios_path(self, dataframe: pd.DataFrame) -> list[str]:
        return dataframe['file_path'].values.tolist()
    
    def load_wav_16k_mono(self, filename: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(filename, normalize = True)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=  True)
        
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq = 16000)
        waveform = resampler(waveform)
        
        waveform = waveform.squeeze()
        return tf.convert_to_tensor(waveform, dtype = tf.float32)
    
    def load_model_mapping(self) -> list[str]:
        class_map_path = "yamnet-model/assets/yamnet_class_map.csv"
        class_names =list(pd.read_csv(class_map_path)['display_name'])

        return class_names

    def get_features(self, dataframe: pd.DataFrame) -> None:
        yamnet_model = self.load_model()
        list_of_paths = self.get_audios_path(dataframe)

        for elem in list_of_paths:
            try:
                wav = self.load_wav_16k_mono(elem)

                # get embeddings
                scores, embeddings, spectrogram = yamnet_model(wav)
            except Exception as e:
                print(f"Error: {e}")
                continue

            return scores, embeddings, spectrogram

    def test_classification(self, dataframe: pd.DataFrame) -> None:
        classes: list[str] = self.load_model_mapping()
        model = self.load_model()
        audio_paths: list[str] = self.get_audios_path(dataframe)

        for elem in audio_paths:
            try:
                scores, embeddings, spectrogram = model(self.load_wav_16k_mono(elem))
                class_scores = tf.reduce_mean(scores, axis=0)
                top_class = tf.argmax(class_scores)
                inferred_class = classes[top_class]
            except Exception as e:
                print(f"Error: {e}")
                continue

            return inferred_class
