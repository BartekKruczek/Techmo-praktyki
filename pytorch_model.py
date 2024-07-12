from torchaudio.models import Wav2Vec2Model

class PytorchModelHandler():
    def __init__(self) -> None:
        # https://pytorch.org/audio/stable/generated/torchaudio.models.Wav2Vec2Model.html#torchaudio.models.Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Pytorch"