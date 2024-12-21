import torch
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, agent_name, ckpt_dir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = agent_name

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_name + '_actor', ckpt_dir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_name + '_critic', ckpt_dir)

        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_name + '_target_actor', ckpt_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_name + '_target_critic', ckpt_dir)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device).flatten()
        actions = self.actor.forward(state)
        noise = torch.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.cpu().detach().numpy()[0]
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()