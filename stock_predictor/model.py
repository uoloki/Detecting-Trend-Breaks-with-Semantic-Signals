# stock_predictor/model.py

import torch
import torch.nn as nn

class BiTALSTM(nn.Module):
    """Bidirectional LSTM with Temporal Attention."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            n_layers,
            dropout=dropout, 
            bidirectional=True, 
            batch_first=True
        )
        hs = hidden_dim * 2  # Hidden size is doubled for bidirectional
        self.Wq = nn.Linear(hs, hs)
        self.Wk = nn.Linear(hs, hs)
        self.v = nn.Linear(hs, 1, bias=False)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        H, _ = self.lstm(x)
        q = self.Wq(H[:, -1])  # Use the last hidden state as the query
        k = self.Wk(H)
        
        # Attention scores
        α = self.v(torch.tanh(k + q.unsqueeze(1)))
        α = torch.softmax(α, dim=1)
        
        # Context vector
        ctx = torch.sum(α * H, dim=1)
        
        # Final prediction
        return self.fc(ctx).squeeze(-1), α.squeeze(-1)


class BiTGRU(nn.Module):
    """Bidirectional GRU with Temporal Attention."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        hs = hidden_dim * 2  # Hidden size is doubled for bidirectional
        self.Wq = nn.Linear(hs, hs)
        self.Wk = nn.Linear(hs, hs)
        self.v = nn.Linear(hs, 1, bias=False)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        H, _ = self.gru(x)
        q = self.Wq(H[:, -1])  # Use the last hidden state as the query
        k = self.Wk(H)

        # Attention scores
        α = self.v(torch.tanh(k + q.unsqueeze(1)))
        α = torch.softmax(α, dim=1)

        # Context vector
        ctx = torch.sum(α * H, dim=1)

        # Final prediction
        return self.fc(ctx).squeeze(-1), α.squeeze(-1)