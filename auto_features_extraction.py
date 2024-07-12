import tensorflow as tf
import pandas as pd
import torchaudio
import torch
import torchaudio.transforms as transforms
import numpy as np
import tensorflow_io as tfio
import librosa

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
    
    @tf.function
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
                wav = self.load_wav_16k_mono(elem)
                print(f"Loaded wav: {wav}")

                scores, embeddings, spectrogram = model(wav)
                print(f"Gotten features: {scores}, {embeddings}, {spectrogram}")
                print(f"Types: {type(scores)}, {type(embeddings)}, {type(spectrogram)}")

                # convert scores to numpy
                scores = np.fromstring(scores)

                print(f"Starting calculating class scores...")
                class_scores = tf.reduce_mean(scores, axis=0)

                print(f"Starting calculating top class...")
                top_class = tf.argmax(class_scores)

                print(f"Starting inferring class...")
                inferred_class = classes[top_class]
            except Exception as e:
                print(f"Error: {e}")
                continue

            return inferred_class
        
    @tf.function
    def load_wav_16k_mono2(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

        return wav

    def test_yamnet_web(self):
        testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')
        testing_wav_data = self.load_wav_16k_mono2(testing_wav_file_name)
        yamnet_model = self.load_model()

        # class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names =list(pd.read_csv("yamnet-model/assets/yamnet_class_map.csv")['display_name'])

        scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(class_scores)
        inferred_class = class_names[top_class]

        print(f'The main sound is: {inferred_class}')
        print(f'The embeddings shape: {embeddings.shape}')
        
