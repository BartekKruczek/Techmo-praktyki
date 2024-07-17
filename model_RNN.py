import torch
import torch.nn as nn

class ModelRNNHandler(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout = 0.5):
        super(ModelRNNHandler, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu RNN z LSTM, biblioteka Pytorch"