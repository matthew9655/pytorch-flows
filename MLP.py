import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, num_qubits, max_params, num_hidden, cond_inputs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_qubits * max_params, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, cond_inputs)

    def forward(self, x):
        x = torch.flatten(x, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x