import pandas as pd

class DataLoaderHandler():
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __repr__(self) -> str:
        return f"Klasa odpowiedzialna za wczytywanie danych z dataframe i ich podział na trzy zbiory"
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe
    
    def split_data(self):
        pass