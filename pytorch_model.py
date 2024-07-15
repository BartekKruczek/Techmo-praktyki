import pandas as pd
import torchaudio
import torch

from transformers import Wav2Vec2Model, Wav2Vec2Processor

class PytorchModelHandler():
    def __init__(self) -> None:
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Pytorch"
    
    def get_audio_path(self, dataframe: pd.DataFrame) -> list[str]:
        return dataframe['file_path'].values.tolist()
    
    def get_audio(self) -> tuple(torch.Tensor, int): # type: ignore
        audio_paths: list[str] = self.get_audio_path()

        for elem in audio_paths:
            try:
                y, sr = torchaudio.load(elem)
            except Exception as e:
                continue

            return y, sr
        
    def extract_features(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        my_inputs = self.processor(audio.squeeze().numpy(), sampling_rate = sr, return_tensors = "pt", padding = True)

        with torch.no_grad():
            features = self.model(input_values = my_inputs.input_values).last_hidden_state
            
        return features.mean(dim = 1)