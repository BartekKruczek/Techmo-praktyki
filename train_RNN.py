import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader_RNN import DataLoaderRNNHandler

class TrainHandlerRNN:
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
        self.writer = SummaryWriter('runs_RNN/experiment1')

    def __repr__(self) -> str:
        return "Klasa odpowiedzialna za trenowanie modelu RNN z LSTM"

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                inputs, labels = data

                inputs = DataLoaderRNNHandler.pad_feature_to_max_dim(self, inputs)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                batch_size, seq_len, feature_dim = inputs.size()
                # print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Feature dimension: {feature_dim}")

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    running_loss = 0.0

            self.writer.add_scalar('training_loss', running_loss / len(self.train_loader), epoch)
            scheduler.step()

            self.model.eval()
            correct = 0
            total = 0
            validation_loss = 0.0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for data in tqdm(self.valid_loader):
                    inputs, labels = data

                    inputs = DataLoaderRNNHandler.pad_feature_to_max_dim(self, inputs)

                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            validation_accuracy = 100 * correct / total
            validation_f1 = f1_score(all_labels, all_predictions, average='weighted')
            validation_precision = precision_score(all_labels, all_predictions, average='weighted')
            validation_recall = recall_score(all_labels, all_predictions, average='weighted')

            tqdm.write(f"Epoch {epoch + 1}, Validation Accuracy: {validation_accuracy:.2f}%, F1 Score: {validation_f1:.2f}, Precision: {validation_precision:.2f}, Recall: {validation_recall:.2f}")
            self.writer.add_scalar('validation_loss', validation_loss / len(self.valid_loader), epoch)
            self.writer.add_scalar('validation_accuracy', validation_accuracy, epoch)
            self.writer.add_scalar('validation_f1', validation_f1, epoch)
            self.writer.add_scalar('validation_precision', validation_precision, epoch)
            self.writer.add_scalar('validation_recall', validation_recall, epoch)

        tqdm.write("Finished Training")

        # test
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                inputs, labels = data

                inputs = DataLoaderRNNHandler.pad_feature_to_max_dim(self, inputs)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_accuracy = 100 * correct / total
        test_f1 = f1_score(all_labels, all_predictions, average='weighted')
        test_precision = precision_score(all_labels, all_predictions, average='weighted')
        test_recall = recall_score(all_labels, all_predictions, average='weighted')

        tqdm.write(f"Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")
        self.writer.add_scalar('test_loss', test_loss / len(self.test_loader), epoch)
        self.writer.add_scalar('test_accuracy', test_accuracy, epoch)
        self.writer.add_scalar('test_f1', test_f1, epoch)
        self.writer.add_scalar('test_precision', test_precision, epoch)
        self.writer.add_scalar('test_recall', test_recall, epoch)

        self.writer.close()
        return test_accuracy
