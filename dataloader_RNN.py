from pandas import DataFrame
from torch import device
from torch._C import device
from dataloader import DataLoaderHandler
from more_features import MoreFeaturesHandler

class DataLoaderRNNHandler(DataLoaderHandler):
    def __init__(self, dataframe: DataFrame, device: device, augmentation: bool, classification: str) -> None:
        super().__init__(dataframe, device, augmentation, classification)
        self.more_features = MoreFeaturesHandler()

    def __repr__(self) -> str:
        return "Klasa do ładowania danych, podejście 2.0 dla RNN"
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int):
        # df oczyszczony, zmienione klasy na liczby
        df = self.dataframe_cleaner()

        audio_path = df.iloc[idx]['file_path']
        label = df.iloc[idx]['healthy_status']

