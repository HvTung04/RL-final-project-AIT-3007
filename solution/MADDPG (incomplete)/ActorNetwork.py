import torch
import torch.nn.functional as F
import torch.nn as nn
import os

class ActorNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir):
        super(ActorNetwork, self).__init__()
        self.ckpt_file = os.path.join(ckpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = torch.softmax(self.pi(x), dim=-1)

        return pi
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.ckpt_file))