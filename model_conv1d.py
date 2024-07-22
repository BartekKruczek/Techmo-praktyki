import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x