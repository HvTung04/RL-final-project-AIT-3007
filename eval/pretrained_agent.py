from torch_model import QNetwork as PretrainNetworkFirst
from final_torch_model import QNetwork as PretrainNetworkFinal
import random
import torch

class RandomPolicy:
  def __init__(self):
    pass
  
  def get_action(self, obs):
    return random.randint(0,20)

class BossPolicy:
    def __init__(self):
        self.q_network = PretrainNetworkFirst((13,13,5), 21)
        self.q_network.load_state_dict(torch.load("red.pt", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    def get_action(self, obs):
        if obs.shape != (13,13,5):
            obs = obs[:,:,:5]
        obs = obs.float().permute([2, 0, 1]).unsqueeze(0)
        q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action

class FinalBossPolicy:
    def __init__(self):
        self.q_network = PretrainNetworkFinal((13,13,5), 21)
        self.q_network.load_state_dict(torch.load("red_final.pt", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.q_network.eval()

    def get_action(self, obs):
        if obs.shape != (13,13,5):
            obs = obs[:,:,:5]
        obs = obs.float().permute([2, 0, 1]).unsqueeze(0)
        q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action