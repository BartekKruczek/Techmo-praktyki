import torch
import tqdm

class TrainHandler():
    def __init__(self, model, X_train, X_test, y_train, y_test, batch_size) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu"
    
    def train(self):
        pass
