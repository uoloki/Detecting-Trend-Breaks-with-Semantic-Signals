import torch
import torch.nn as nn
import math

class CNNLSTMAttention(nn.Module):
    """CNN feature extractor + LSTM + Attention for temporal patterns."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # CNN layers for local feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1)
        self.dropout_conv = nn.Dropout(dropout)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            hidden_dim//2, hidden_dim, n_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        
        # Attention mechanism
        hs = hidden_dim * 2
        self.Wq = nn.Linear(hs, hs)
        self.Wk = nn.Linear(hs, hs)
        self.v = nn.Linear(hs, 1, bias=False)
        self.fc = nn.Linear(hs, 1)
        
    def forward(self, x):
        # CNN feature extraction
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = torch.relu(self.conv1(x_conv))
        x_conv = self.dropout_conv(x_conv)
        x_conv = torch.relu(self.conv2(x_conv))
        x_conv = self.dropout_conv(x_conv)
        
        # Back to (batch, seq_len, features)
        x_conv = x_conv.transpose(1, 2)
        
        # LSTM processing
        H, _ = self.lstm(x_conv)
        
        # Attention
        q = self.Wq(H[:, -1])
        k = self.Wk(H)
        α = self.v(torch.tanh(k + q.unsqueeze(1)))
        α = torch.softmax(α, dim=1)
        ctx = torch.sum(α * H, dim=1)
        
        return self.fc(ctx).squeeze(-1), α.squeeze(-1)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerPredictor(nn.Module):
    """Transformer-based sequence predictor."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, n_heads=8):
        super().__init__()
        
        # Ensure hidden_dim is divisible by n_heads
        if hidden_dim % n_heads != 0:
            hidden_dim = ((hidden_dim // n_heads) + 1) * n_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        # Note: We don't need attention mask for this case
        encoded = self.transformer(x)
        
        # Global pooling and prediction
        # encoded: (batch, seq_len, hidden_dim) -> (batch, hidden_dim, seq_len)
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        output = self.fc(self.dropout(pooled)).squeeze(-1)
        
        # Return output and dummy attention weights for compatibility
        return output, torch.ones(x.size(0), x.size(1), device=x.device)

class HybridCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer model."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # CNN for local patterns
        self.conv1 = nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5, padding=2)
        
        # Transformer for global patterns
        self.input_projection = nn.Linear(hidden_dim//2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, n_layers//2))
        
        # Output
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN processing
        x_conv = x.transpose(1, 2)
        x_conv = torch.relu(self.conv1(x_conv))
        x_conv = torch.relu(self.conv2(x_conv))
        x_conv = x_conv.transpose(1, 2)
        
        # Transformer processing
        x_trans = self.input_projection(x_conv)
        x_trans = self.transformer(x_trans)
        
        # Final prediction (use last timestep)
        output = self.fc(self.dropout(x_trans[:, -1])).squeeze(-1)
        
        return output, torch.ones(x.size(0), x.size(1), device=x.device)

class ResidualLSTM(nn.Module):
    """LSTM with residual connections and layer normalization."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection to match hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multiple LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(n_layers):
            # First layer takes projected input, rest take hidden_dim * 2 from bidirectional
            input_size = hidden_dim if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(input_size, hidden_dim, 1, dropout=dropout, 
                       bidirectional=True, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * 2))
        
        # Attention and output
        hs = hidden_dim * 2
        self.Wq = nn.Linear(hs, hs)
        self.Wk = nn.Linear(hs, hs)
        self.v = nn.Linear(hs, 1, bias=False)
        self.fc = nn.Linear(hs, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Process through LSTM layers with residuals
        residual = None
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            H, _ = lstm(x)
            H = norm(H)
            
            if residual is not None and H.shape == residual.shape:
                # Add residual connection only if shapes match
                H = H + residual
            
            residual = H
            x = H  # Input for next layer
        
        # Attention mechanism
        q = self.Wq(H[:, -1])
        k = self.Wk(H)
        α = self.v(torch.tanh(k + q.unsqueeze(1)))
        α = torch.softmax(α, dim=1)
        ctx = torch.sum(α * H, dim=1)
        
        return self.fc(self.dropout(ctx)).squeeze(-1), α.squeeze(-1) 