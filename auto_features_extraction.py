import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

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
        file_contents = tf.io.read_file(file_path)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

        return wav
