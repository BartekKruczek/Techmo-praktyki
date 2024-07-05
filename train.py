import torch
import tqdm

class TrainHandler():
    def __init__(self, model, X_train, X_test, y_train, y_test, batch_size, learning_rate, num_epochs) -> None:
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu"
    
    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            for i in tqdm.tqdm(range(0, len(self.X_train), self.batch_size)):
                X_batch = self.X_train[i:i+self.batch_size].view(-1, 1, 13, 13)
                y_batch = self.y_train[i:i+self.batch_size]

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i in tqdm.tqdm(range(0, len(self.X_test), self.batch_size)):
                    X_batch = self.X_test[i:i+self.batch_size].view(-1, 1, 13, 13)
                    y_batch = self.y_test[i:i+self.batch_size]

                    output = self.model(X_batch)
                    _, predicted = torch.max(output, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}, Accuracy: {round(correct/total, 3)}")
        return self.model
