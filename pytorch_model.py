import pandas as pd
import torchaudio
import torch

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchaudio.transforms import Resample

class PytorchModelHandler():
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.dataframe = dataframe

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Pytorch"
    
    def get_audio_paths(self) -> list[str]:
        return self.dataframe['file_path'].values.tolist()
    
    def get_audio(self, file_path) -> tuple[torch.Tensor, int]:
        try:
            y, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return None, None
        return y, sr
    
    def change_sr(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        return Resample(orig_freq = sr, new_freq = 16000)(audio)
        
    def extract_features(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        my_inputs = self.processor(audio.squeeze().numpy(), sampling_rate = sr, return_tensors = "pt", padding = True)

        with torch.no_grad():
            features = self.model(input_values = my_inputs.input_values).last_hidden_state
        return features.mean(dim = 1)
    
    def get_features(self) -> torch.Tensor:
        list_of_audio_paths = self.get_audio_paths()

        for elem in list_of_audio_paths:
            audio, sr = self.get_audio(elem)
            if sr != 16000:
                audio = self.change_sr(audio, sr)

            features = self.extract_features(audio, 16000)
            return features