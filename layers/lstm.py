import torch
import torch.nn as nn

class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, num_layers, dropout=0.1):
        super(LSTMTimeSeriesModel, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Output projection to map back to input dimension
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        
        # Pass through LSTM
        # h0, c0: initial hidden and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_len, hidden_dim)
        
        # Project back to input_dim
        out = self.output_projection(lstm_out)
        
        return out

# # Model parameters
# input_dim = 128  # Each grid point has 128 features
# seq_len = 1052  # Number of grid points
# hidden_dim = 256  # Hidden dimension of LSTM
# num_layers = 2  # Number of LSTM layers
# dropout = 0.1  # Dropout rate

# # Instantiate the model
# model = LSTMTimeSeriesModel(input_dim, seq_len, hidden_dim, num_layers, dropout)

# # Example input
# batch_size = 32
# x = torch.randn(batch_size, seq_len, input_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# # Forward pass
# output = model(x)
# print("Output shape:", output.shape)  # Expected output shape: (batch_size, seq_len, input_dim)
