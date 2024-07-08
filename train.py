import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class TrainHandler():
    def __init__(self, model, train_set, valid_set, test_set, device) -> None:
        self.model = model.to(device)
        self.train_loader = train_set
        self.valid_loader = valid_set
        self.test_loader = test_set
        self.device = device
        self.learning_rate = 0.001
        self.num_epochs = 5

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu"

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Reshape the inputs to add the channel dimension (batch_size, 1, height, width)
                inputs = inputs.unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    tqdm.write(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.valid_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.unsqueeze(1)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            tqdm.write(f"Epoch {epoch + 1}, Validation Accuracy: {100 * correct / total:.2f}%")

        tqdm.write("Finished Training")

        # Testing
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.unsqueeze(1)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        tqdm.write(f"Test Accuracy: {100 * correct / total:.2f}%")
