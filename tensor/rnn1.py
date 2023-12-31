import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

input_size = 10   # Number of input features
hidden_size = 20  # Number of features in the hidden state
output_size = 5   # Number of output features

model = SimpleRNN(input_size, hidden_size, output_size)

batch_size = 1
seq_length = 3  # Length of the sequence

dummy_input = torch.randn(batch_size, seq_length, input_size)

output = model(dummy_input)
print(output)
