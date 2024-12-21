from final_torch_model import QNetwork
import torch

class BOSS_DQN:
    def __init__(self):
        self.q_network = QNetwork((13,13,5), 21)
        self.q_network.load_state_dict(torch.load("red_final.pt", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.q_network.eval()

    def get_action(self, obs):
        if obs.shape != (13,13,5):
            obs = obs[:,:,:5]
        obs = obs.float().permute([2, 0, 1]).unsqueeze(0)
        q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action