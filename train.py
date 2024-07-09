import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class TrainHandler():
    def __init__(self, model, train_set, valid_set, test_set, device, learning_rate, num_epochs, step_size, gamma, l1_lambda, l2_lambda) -> None:
        self.model = model.to(device)
        self.train_loader = train_set
        self.valid_loader = valid_set
        self.test_loader = test_set
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu"

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                inputs = inputs.unsqueeze(1)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # L1
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    tqdm.write(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

            scheduler.step()

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

        # test
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

        test_accuracy = 100 * correct / total
        tqdm.write(f"Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy
