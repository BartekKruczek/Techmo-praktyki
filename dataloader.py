import pandas as pd
import numpy as np

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

        X = dataframe['MFCC']
        X = np.asarray(X)
        print(np.shape(X))
        y = dataframe['healthy_status'].values
        y = np.asarray(y)

        # split data into three sets: train, validation, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # if type(X_train) != np.ndarray:
        #     X_train = np.array(X_train)

        # if type(X_test) != np.ndarray:
        #     X_test = np.array(X_test)

        # if type(y_train) != np.ndarray:
        #     y_train = np.array(y_train)

        # if type(y_test) != np.ndarray:
        #     y_test = np.array(y_test)

        print(f"X_train type: {type(X_train)}")
        print(f"X_test type: {type(X_test)}")
        print(f"y_train type: {type(y_train)}")
        print(f"y_test type: {type(y_test)}")

        return X_train, X_test, y_train, y_test
