import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import torchaudio
import torch
import torchaudio.transforms as transforms

class AutoFeaturesExtraction:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do automatycznego ekstrahowania cech, YAMNet"

    def load_model(self) -> tf.keras.Model:
        yamnet_model_handle = 'https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)

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

    def get_features(self, dataframe: pd.DataFrame) -> None:
        yamnet_model = self.load_model()
        list_of_paths = self.get_audios_path(dataframe)

        print(f"Starting extracting features...")
        for elem in list_of_paths:
            try:
                wav = self.load_wav_16k_mono(elem)

                print(f"Processing file {elem}")
                # get embeddings
                scores, embeddings, spectrogram = yamnet_model(wav)
                print(f"Processed file {elem}")
            except Exception as e:
                print(f"Error: {e}")
                continue
        print(f"Done!")

    def test(self) -> None:
        testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')
        testing_wav_data = self.load_wav_16k_mono(testing_wav_file_name)

        yamnet_model = self.load_model()
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names =list(pd.read_csv(class_map_path)['display_name'])

        scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(class_scores)
        inferred_class = class_names[top_class]

        print(f'The main sound is: {inferred_class}')
        print(f'The embeddings shape: {embeddings.shape}')
