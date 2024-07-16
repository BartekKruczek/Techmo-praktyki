from pandas import DataFrame
from torch import device
from torch._C import device
from dataloader import DataLoaderHandler

class DataLoaderRNNHandler(DataLoaderHandler):
    def __init__(self, dataframe: DataFrame, device: device, augmentation: bool, classification: str) -> None:
        super().__init__(dataframe, device, augmentation, classification)

    def __repr__(self) -> str:
        return "Klasa do ładowania danych, podejście 2.0 dla RNN"
    
    def __len__(self) -> int:
        return len(self.dataframe)