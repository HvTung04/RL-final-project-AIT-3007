from magent2.environments import battle_v4
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
%matplotlib inline

def get_team_observations(env, handle):
    """
    Get team observations from environment and convert to tensor.

    Args:
        env: Environment object
        handle: Team handle (e.g. "red", "blue")

    Returns:
        observations: List of team observations size N x 13 x 13 x 9"""
    observations = []
    for agent in env.possible_agents:
        if handle in agent:
            if env.terminations.get(agent, True) or env.truncations.get(agent, True):
                obs = np.zeros((13, 13, 9))
            else:
                obs = env.observe(agent)
            observations.append(torch.tensor(obs, dtype=torch.float32))

    return observations

def get_agent_positions(team_obs):
    """
    Extract positions of all agents from team observations.

    Args:
        team_obs: List of team observations.
                 Each obs shape is (13, 13, 9) where:
                 - channels[-2:] are the agent's absolute position (x, y)

    Returns:
        Dict of agent positions {agent_idx: (x, y)}
    """
    positions = {}

    # For each agent in the team
    for agent_idx, obs in enumerate(team_obs):
        # Get absolute positions from last two channels
        x_pos = obs[0, 0, -2]  # Second to last channel for x coordinate
        y_pos = obs[0, 0, -1]  # Last channel for y coordinate

        positions[agent_idx] = (x_pos, y_pos)

    return positions

def find_dead(env, handle):
    """
    Find dead agents in the environment.

    Args:
        env: Environment object
        handle: Team handle (e.g. "red", "blue")

    Returns:
        List of dead agents
    """
    dead_agents = []
    for agent in env.possible_agents:
        if handle in agent:
            if env.terminations.get(agent, True) or env.truncations.get(agent, True):
                agent_id = int(agent.split('_')[1])
                dead_agents.append(agent_id)
    return dead_agents

def construct_adjacency_matrix(team_obs, n_neighbors=4):
    """
    Construct adjacency matrix C^t_i for each agent i.

    Args: Dict of team observations {agent_idx: obs}
          n_neighbors: Number of neighbors to consider (default: 4)

    Returns: List of adjacency matrices for each agent
    """
    adjacency_matrices = []

    # Get agent positions
    positions = get_agent_positions(team_obs)
    num_agents = 81

    for i in positions.keys():
        distances = []
        for j in positions.keys():
            if i != j:
                # Calculate Euclidean distance between agent i and j (in normalized coordinates)
                dist = ((positions[j][0] - positions[i][0]) ** 2 + (positions[j][1] - positions[i][1]) ** 2) ** 0.5
                distances.append((dist, j))

        distances.sort(key=lambda x: x[0])  # Sort by distance

        adj_matrix = torch.zeros((n_neighbors + 1, num_agents))

        adj_matrix[0, i] = 1  # Self-connection

        for idx, (dist, neighbor_idx) in enumerate(distances[:n_neighbors]):
            adj_matrix[idx + 1, neighbor_idx] = 1

        while(len(distances) < n_neighbors + 1):
            adj_matrix.append(torch.zeros((n_neighbors + 1, num_agents)))

        adjacency_matrices.append(adj_matrix)

    # Pad with zero matrices if needed
    while len(adjacency_matrices) < num_agents:
        adjacency_matrices.append(torch.zeros((n_neighbors + 1, num_agents)))

    return adjacency_matrices

def process_timestep(env, handle):
    """
    Process one timestep for all agents.

    Args:
        env: Environment object
        handle: Team handle (e.g. "red", "blue")

    Returns:
        observations: List of size N with each element as a tensor of shape (13, 13, 9)
        adjacency_matrices: List of size N with each element as a tensor of shape (5, N)
    """
    # Get team observations
    observations = get_team_observations(env, handle)

    # Get adjacency matrices for all agents
    adjacency_matrices = construct_adjacency_matrix(observations)

    return observations, adjacency_matrices

# Multihead dot-product attention as the convolutional kernel => interaction between agents

class MultiheadAttention(nn.Module):
    """
    Multihead attention module for DGN

    Args:
        n_neighbors: Number of neighbors to consider
        input_dim: Input dimension of the feature vectors
        head_dim: Dimension of each head
        num_heads: Number of attention heads
        output_dim: Output dimension of the feature vectors
    """
    def __init__(self, n_neighbors, input_dim, head_dim, num_heads, output_dim):
        super(MultiheadAttention, self).__init__()

        self.n_neighbors = n_neighbors
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        # Linear transformation for Q, K, V
        self.query_layer = nn.Linear(input_dim, head_dim * num_heads)
        self.key_layer = nn.Linear(input_dim, head_dim * num_heads)
        self.value_layer = nn.Linear(input_dim, head_dim * num_heads)

        # Final transformation
        self.output_layer = nn.Linear(head_dim * num_heads, output_dim)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, query, value, key, extra_vector):
        """
        Forward pass for multihead attention

        Args:
            query: Query tensor of shape (batch_size, n_neighbors + 1, L = input_dim)
            value: Value tensor of shape (batch_size, n_neighbors + 1, L = input_dim)
            key: Key tensor of shape (batch_size, n_neighbors + 1, L = input_dim)
            extra_vector: Extra vector of shape (batch_size, 1, n_neighbors + 1)

        Returns:
            Output tensor of shape (batch_size, 1, output_dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            extra_vector = extra_vector.unsqueeze(0)

        batch_size = query.size(0)

        # Linear transformation with ReLU
        query_2 = self.relu(self.query_layer(query)) # Shape (batch_size, n_neighbors + 1, head_dim * num_heads)
        key_2 = self.relu(self.key_layer(key))  # Shape (batch_size, n_neighbors + 1, head_dim * num_heads)
        value_2 = self.relu(self.value_layer(value))    # Shape (batch_size, n_neighbors + 1, head_dim * num_heads)

        # Reshape to separate heads
        # Shape (batch_size, n_neighbors + 1, num_heads, head_dim)
        reshaped_query = query_2.view(batch_size, self.n_neighbors + 1, self.num_heads, self.head_dim)
        reshaped_key = key_2.view(batch_size, self.n_neighbors + 1, self.num_heads, self.head_dim)
        reshaped_value = value_2.view(batch_size, self.n_neighbors + 1, self.num_heads, self.head_dim)

        # Permute dimensions for attention
        # Shape (batch_size, num_heads, n_neighbors + 1, head_dim)
        final_query = reshaped_query.permute(0, 2, 1, 3)
        final_key = reshaped_key.permute(0, 2, 1, 3)
        final_value = reshaped_value.permute(0, 2, 1, 3)


        # Scaled dot-product attention
        attention_scores = torch.matmul(final_query, final_key.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (batch_size, num_heads, n_neighbors + 1, n_neighbors + 1)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, n_neighbors + 1, n_neighbors + 1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, final_value)  # (batch_size, num_heads, n_neighbors + 1, head_dim)

        # Permute back and reshape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, n_neighbors + 1, num_heads, head_dim)
        attention_output = attention_output.view(batch_size, self.n_neighbors + 1, -1)  # (batch_size, n_neighbors + 1, num_heads * head_dim)

        # Apply extra vector
        output = torch.matmul(extra_vector, attention_output)  # (batch_size, 1, num_heads * head_dim)

        # Final linear transformation
        output = self.relu(self.output_layer(output))  # (batch_size, 1, output_dim)

        return output

class QNetwork(nn.Module):
    def __init__(self, feature_dim=128, action_dim=21):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, feature, relation1, relation2):
        combined = torch.cat([feature, relation1, relation2], dim=-1)
        return self.network(combined)


# MLP for observation encoding

class MLP(nn.Module):
    def __init__(self, observation_shape=(13, 13, 9), feature_dim=128):
        super(MLP, self).__init__()

        # Calculate the flattened input size
        self.input_dim = observation_shape[0] * observation_shape[1] * observation_shape[2]  # 13 * 13 * 9

        # Define the network layers
        self.layers = nn.Sequential(
            # First flatten the input
            nn.Flatten(),

            # First dense layer with 512 units
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),

            # Second dense layer with feature_dim units (128 in original)
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )

        # Reshape layer to match original output shape (1, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # x shape: (13, 13, 9)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # x shape: (batch_size, 13, 13, 9)

        x = x.permute(0, 3, 1, 2)

        # Pass through MLP layers
        features = self.layers(x)

        # Reshape to (batch_size, 1, feature_dim) to match original output shape
        features = features.unsqueeze(1)

        return features
    
class DGN(nn.Module):
    def __init__(self, observation_shape=(13, 13, 9), feature_dim=128, action_dim=21):
        super(DGN, self).__init__()

        self.feature_dim = feature_dim

        self.mlp = MLP(observation_shape, feature_dim)

        self.attention1 = MultiheadAttention(n_neighbors=4, input_dim=feature_dim, head_dim=16, num_heads=8, output_dim=feature_dim)
        self.attention2 = MultiheadAttention(n_neighbors=4, input_dim=feature_dim, head_dim=16, num_heads=8, output_dim=feature_dim)

        self.q_network = QNetwork(feature_dim, action_dim)

        # Initialize extra_vector with batch dimension
        self.extra_vector_template = nn.Parameter(torch.zeros(1, 1, 5))
        nn.init.ones_(self.extra_vector_template)
        self.extra_vector_template.data[0, 0, 0] = 1

    def local_features(self, features, adjacency_matrices):
        """
        Compute local features for all agents

        Args:
            features: Tensor of shape (batch_size, N, feature_dim)
            adjacency_matrices: Tensor of shape (batch_size, n_neighbors + 1, N)

        Returns:
            local_features: Tensor of shape (batch_size, n_neighbors + 1, feature_dim)
        """
        local_features = torch.matmul(adjacency_matrices, features)

        return local_features

    def forward(self, team_observations, adjacency_matrices):
        """
        Forward pass for DGN

        Args:
            team_observations: List N Tensor of shape (batch_size, 13, 13, 9)
            adjacency_matrices: List N Tensor of shape (batch_size, n_neighbors + 1, N)

        Returns:
            q_values: Q values of shape (batch_size, action_dim)
        """
        n_agents = 81
        if team_observations[0].dim() == 3:
            team_observations = [obs.unsqueeze(0) for obs in team_observations]
        batch_size = team_observations[0].size(0)
        extra_vector = self.extra_vector_template.expand(batch_size, -1, -1)

        # Etract features for all agents
        team_features = torch.cat([self.mlp(obs) for obs in team_observations], dim=1)  # Shape (batch_size, N, feature_dim)

        # First relation layer
        local_features = [self.local_features(team_features, adjacency_matrix) for adjacency_matrix in adjacency_matrices]  # Shape (batch_size, n_neighbors + 1, feature_dim)
        relation1 = {}
        for i in range(n_agents):
            query = key = value = local_features[i]
            relation1[i] = self.attention1(query, value, key, extra_vector) # Shape (batch_size, 1, feature_dim)

        # Second relation layer
        relation2 = {}
        for i in range(n_agents):
            start_index = max(0, i - 4 // 2)
            end_index = min(n_agents, start_index + 4)

            start_index = max(0, end_index - 4)
            neighbors = list(relation1.values())[start_index:end_index]
            neighbors.append(relation1[i])
            if len(neighbors) < 5:
                for _ in range(5-len(neighbors)):
                    neighbors.append(torch.zeros(relation1[i].shape))
            neighbors = [neighbor.to(device) for neighbor in neighbors]

            query = torch.cat(neighbors, dim=1).to(device)
            key = torch.cat(neighbors, dim=1).to(device)
            value = torch.cat(neighbors, dim=1).to(device)
            # Process attention
            rel2 = self.attention2(query, key, value, extra_vector)
            relation2[i] = rel2 # Shape (batch_size, 1, feature_dim)

        # Compute Q values
        q_values = []
        for i in range(n_agents):
            agent_q = self.q_network(team_features[:, i, :].unsqueeze(1), relation1[i], relation2[i]) # Shape (batch_size, 1, action_dim)
            q_values.append(agent_q)

        # Stack Q values to create Q table
        q_values = torch.cat(q_values, dim=1) # Shape (batch_size, N, action_dim)

        return q_values

blue_team = DGN()  
blue_team.load_state_dict(torch.load('blue_team_model.pth')) # Load the saved state dictionary
blue_team.eval()

class BLUE_DGN:
    def __init__(self):
        self.blue_team = blue_team

    def get_action(self, team_obs, team_adj):
        """
        Get actions of team
        """
        if obs.shape != (13,13,9):
            obs = obs[:,:,:9]
        obs = obs.float().permute([2, 0, 1]).unsqueeze(0)
        q_values = self.blue_team(obs, adj)
        action = torch.argmax(q_values, dim=2).numpy()[0]
        return action