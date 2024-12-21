import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, handle, batch_size):
        self.mem_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        self.handle = handle

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))

        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims)))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims)))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_counter % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[f"{self.handle}_{agent_idx}"].flatten()
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[f"{self.handle}_{agent_idx}"].flatten()
            self.actor_action_memory[agent_idx][index] = action[f"{self.handle}_{agent_idx}"]
        
        self.state_memory[index] = state.flatten()
        self.new_state_memory[index] = state_.flatten()
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1
    
    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
        
        return actor_states, states, actions, rewards, actor_new_states, states_, terminal
    
    def ready(self):
        if self.mem_counter >= self.batch_size:
            return True
        return False