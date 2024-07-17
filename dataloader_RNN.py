import torch
from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
from dataloader import DataLoaderHandler
from more_features import MoreFeaturesHandler
from pytorch_model import PytorchModelHandler

class DataLoaderRNNHandler(DataLoaderHandler, PytorchModelHandler, MoreFeaturesHandler):
    def __init__(self, dataframe: DataFrame, device: torch.device, augmentation: bool, classification: str) -> None:
        DataLoaderHandler.__init__(self, dataframe, device, augmentation, classification)
        PytorchModelHandler.__init__(self, dataframe, classification)
        MoreFeaturesHandler.__init__(self)

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
    
    @staticmethod
    def pad_feature_to_max_dim(self, input, max_dim = 2500) -> torch.Tensor:
        batch_size, seq_len, feature_dim = input.size()

        if feature_dim > max_dim:
            input = input[:, :, :max_dim]
        elif feature_dim < max_dim:
            padding = torch.zeros(batch_size, seq_len, max_dim - feature_dim, dtype=input.dtype)
            input = torch.cat((input, padding), dim=2)
        return input
    
    def check_inputs_shape(self, train_loader) -> None:
        max_dim = 0
        all_values = []
        all_counter = 0
        counter_greater_than_3000 = 0
        counter_greater_than_2000 = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            batch_size, seq_len, feature_dim = inputs.size()
            all_values.append(feature_dim)
            all_counter += 1
            print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Feature dimension: {feature_dim}")

            if feature_dim > max_dim:
                max_dim = feature_dim

            if feature_dim > 3000:
                counter_greater_than_3000 += 1

            if feature_dim > 2000:
                counter_greater_than_2000 += 1

        print(f"Max feature dimension: {max_dim}")
        print(f"Mean feature dimension: {sum(all_values) / len(all_values)}")
        print(f"Counter greater than 3000: {counter_greater_than_3000}")
        print(f"Counter greater than 2000: {counter_greater_than_2000}")
        print(f"Counter all: {all_counter}")

    def check_inputs_shape_after_padding(self, train_loader) -> None:
        max_dim = 0
        all_values = []
        all_counter = 0
        counter_greater_than_3000 = 0
        counter_greater_than_2000 = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = self.pad_feature_to_max_dim(inputs)
            batch_size, seq_len, feature_dim = inputs.size()
            all_values.append(feature_dim)
            all_counter += 1
            print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Feature dimension: {feature_dim}")

            if feature_dim > max_dim:
                max_dim = feature_dim

            if feature_dim > 3000:
                counter_greater_than_3000 += 1

            if feature_dim > 2000:
                counter_greater_than_2000 += 1

        print(f"Max feature dimension: {max_dim}")
        print(f"Mean feature dimension: {sum(all_values) / len(all_values)}")
        print(f"Counter greater than 3000: {counter_greater_than_3000}")
        print(f"Counter greater than 2000: {counter_greater_than_2000}")
        print(f"Counter all: {all_counter}")