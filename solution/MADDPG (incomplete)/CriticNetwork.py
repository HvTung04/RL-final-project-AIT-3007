import torch
import torch.nn.functional as F
import torch.nn as nn
import os

class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, ckpt_dir):
        super(CriticNetwork, self).__init__()
        self.ckpt_file = os.path.join(ckpt_dir, name)
        self.n_agents = n_agents

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        cat_input = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(cat_input))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.ckpt_file))