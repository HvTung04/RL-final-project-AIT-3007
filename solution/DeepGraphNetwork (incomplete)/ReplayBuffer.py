from collections import deque
import torch
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        batch = []
        if self.num_experiences < batch_size:
            batch = random.sample(self.buffer, self.num_experiences)
        else:
            batch = random.sample(self.buffer, batch_size)
                # Initialize lists to store the stacked elements
        states, actions, rewards, next_states, dones, adjacencies = [], [], [], [], [], []

        # Iterate over the sampled batch and append the corresponding elements
        for experience in batch:
            state, action, reward, next_state, done, adjacency = experience
            states.append(torch.stack(state)) # Shape (N, 13, 13, 9)
            actions.append(torch.tensor([a if a is not None else 0 for a in action])) # Shape (N)
            rewards.append(torch.tensor([r if r is not None else 0 for r in reward])) # Shape (N)
            next_states.append(torch.stack(next_state)) # Shape (N, 13, 13, 9)
            dones.append(done) # Shape (N)
            adjacencies.append(torch.stack(adjacency)) # Shape (N, 5, N)

        # Stack the elements in each list
        batched_states = torch.stack(states).permute(1, 0, 2, 3, 4) # Shape (B, N, 13, 13, 9) -> Shape (N, B, 13, 13, 9)
        batched_actions = torch.stack(actions).permute(1, 0) # Shape (B, N) -> Shape (N, B)
        batched_rewards = torch.stack(rewards).permute(1, 0) # Shape (B, N) -> Shape (N, B)
        batched_next_states = torch.stack(next_states).permute(1, 0, 2, 3, 4) # Shape (B, N, 13, 13, 9) -> Shape (N, B, 13, 13, 9) 
        batched_dones = torch.stack(dones).permute(1,0) # Shape (B, N) -> Shape (N, B)
        batched_adjacencies = torch.stack(adjacencies).permute(1, 0, 2, 3) # Shape (B, N, 5, N)

        return (batched_states, batched_actions, batched_rewards, batched_next_states, batched_dones, batched_adjacencies)
    
    def size(self):
        return self.buffer_size 
    
    def add(self, state, action, reward, next_state, done, adjacency):
        experience = (state, action, reward, next_state, done, adjacency)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def count(self):
        return self.num_experiences
    
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
        