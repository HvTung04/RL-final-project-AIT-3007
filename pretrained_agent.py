from torch_model import QNetwork
import torch

class NOOB_DQN:
    def __init__(self):
        self.q_network = QNetwork((13,13,5), 21)
        self.q_network.load_state_dict(torch.load("red.pt", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    def get_action(self, obs):
        if obs.shape != (13,13,5):
            obs = obs[:,:,:5]
        obs = obs.float().permute([2, 0, 1]).unsqueeze(0)
        q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action

