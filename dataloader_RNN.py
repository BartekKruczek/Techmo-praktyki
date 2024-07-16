import torch

from pandas import DataFrame
from dataloader import DataLoaderHandler
from more_features import MoreFeaturesHandler
from pytorch_model import PytorchModelHandler

class DataLoaderRNNHandler(DataLoaderHandler, PytorchModelHandler):
    def __init__(self, dataframe: DataFrame, device: torch.device, augmentation: bool, classification: str) -> None:
        DataLoaderHandler.__init__(self, dataframe, device, augmentation, classification)
        PytorchModelHandler.__init__(self, dataframe, classification)
        self.more_features = MoreFeaturesHandler()

    def __repr__(self) -> str:
        return "Klasa do ładowania danych, podejście 2.0 dla RNN"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int):
            features, label = PytorchModelHandler.__getitem__(self, idx)
            
            features = torch.tensor(features, dtype=torch.float32)
            features = features.squeeze(0)

            label = torch.tensor(label, dtype=torch.long)
            return features, label
    
    def pad_features(self, features, pad_size):
        if features.size(1) > pad_size:
            return features[:, :pad_size]
        elif features.size(1) < pad_size:
            pad = torch.zeros(features.size(0), pad_size - features.size(1))
            return torch.cat((features, pad), dim=1)
        else:
            return features
    
    def check_inputs_shape(self, train_loader) -> None:
        for i, data in enumerate(train_loader):
            inputs, labels = data
            batch_size, seq_len, feature_dim = inputs.size()
            print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Feature dimension: {feature_dim}")