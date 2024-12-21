import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_shape, actions_dim, kernel_size=3, stride=1, padding=1):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            obs_shape[-1], 32, kernel_size=kernel_size, stride=stride, padding=padding
        )  # 5 channels -> 32 features map
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )  # 32 channels -> 64 features map

        self.fc1 = nn.Linear(
            64 * obs_shape[0] * obs_shape[1], 128
        )  # Flat 64 * 13 * 13 -> 128 neurons
        self.fc2 = nn.Linear(128, actions_dim)  # 128 neurons -> 1 action

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        x = x.permute(
            0, 3, 1, 2
        )  # (batch_size, width, height, channels) -> (batch_size, channels, width, height)

        # Convolution with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = x.reshape(x.size(0), -1)  # (batch_size, flatten_size)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_shape, actions_dim):
        super(DuelingQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(obs_shape[-1], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * obs_shape[0] * obs_shape[1], 128)
        
        # Separate streams for value and advantage
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, actions_dim)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)  # (batch_size, width, height, channels) -> (batch_size, channels, width, height)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.reshape(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DoubleQPolicy:
    def __init__(self, pretrained_path):
        self.q_network = QNetwork((13,13,5), 21)
        self.q_network.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.q_network.eval()

    def get_action(self, obs):
        q_values = self.q_network(obs) # batch_size x actions_dim
        action = torch.argmax(q_values, dim=1).unsqueeze(0)
        return action.numpy()[0].item()
    
class DuelingQPolicy:
    def __init__(self, pretrained_path):
        self.q_network = DuelingQNetwork((13,13,5), 21)
        self.q_network.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.q_network.eval()

    def get_action(self, obs):
        q_values = self.q_network(obs) # batch_size x actions_dim
        action = torch.argmax(q_values, dim=1).unsqueeze(0)
        return action.numpy()[0].item()