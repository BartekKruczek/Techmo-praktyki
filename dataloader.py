import pandas as pd

from sklearn.model_selection import train_test_split

class DataLoaderHandler():
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __repr__(self) -> str:
        return f"Klasa odpowiedzialna za wczytywanie danych z dataframe i ich podział na trzy zbiory"
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe
    
    def split_data(self):
        dataframe = self.get_dataframe()

        # split data into three sets: train, validation, test
        X_train, X_test, y_train, y_test = train_test_split(dataframe['MFCC'], dataframe['healthy_status'], test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test