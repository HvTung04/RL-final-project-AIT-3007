from Agent import Agent
import torch
import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, handle, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, ckpt_dir="tmp/maddpg"):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        for agent_idx in range(self.n_agents):
            self.agents.append(
                Agent(actor_dims, critic_dims, n_agents, n_actions, f"{handle}_{agent_idx}", ckpt_dir)
            )

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        for agent in self.agents:
            actions[agent.agent_name] = agent.choose_action(raw_obs[agent.agent_name])
        
        return actions
    
    def learn(self, memory):
        if not memory.ready():
            return
        
        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agent_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx], dtype=torch.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agent_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1).to(device)
        mu = torch.cat([acts for acts in all_agent_new_mu_actions], dim=1).to(device)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1).to(device)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            # actions = actions.unsqueeze(0).expand(len(self.agents), -1, -1)
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()