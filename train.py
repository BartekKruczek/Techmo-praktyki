import torch

class TrainHandler():
    def __init__(self, model, X_train, X_test, y_train, y_test) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 10

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu"
    
    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = criterion(output, self.y_train)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                output = self.model(self.X_test)
                _, predicted = torch.max(output, 1)
                total += self.y_test.size(0)
                correct += (predicted == self.y_test).sum().item()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}, Accuracy: {round(correct/total, 3)}")
        return self.model
