import pandas as pd

from transformers import Wav2Vec2Model, Wav2Vec2Processor

class PytorchModelHandler():
    def __init__(self) -> None:
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Pytorch"
    
    def get_audio_path(self, dataframe: pd.DataFrame) -> list[str]:
        return dataframe['file_path'].values.tolist()
    
    def get_feature(self) -> None:
        pass