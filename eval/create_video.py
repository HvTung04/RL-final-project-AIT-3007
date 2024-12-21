from magent2.environments import battle_v4
import os
import cv2
from pretrained_agent import RandomPolicy, BossPolicy, FinalBossPolicy
from self_trained_agent import DoubleQPolicy, DuelingQPolicy
import torch

env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles=300)

vid_dir = "video15"
os.makedirs(vid_dir, exist_ok=True)
fps = 15
frames = []

random_red = RandomPolicy()
pretrained_red = BossPolicy()
boss_red = FinalBossPolicy()

self_blue = DoubleQPolicy("blue_deepQ.pth")

def create_video(red_team, blue_team, video_title):
    frames = []
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            agent_handle = agent.split("_")[0]
            observation = torch.tensor(observation)
            if agent_handle == "red":
                with torch.no_grad():
                    action = red_team.get_action(observation)
            else:
                with torch.no_grad():
                    action = blue_team.get_action(observation)
        env.step(action)

        if env.agents and agent == env.agents[-1]:
            frames.append(env.render())
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"{video_title}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    ) 

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

create_video(random_red, self_blue, "random")
create_video(pretrained_red, self_blue, "pretrained")
create_video(boss_red, self_blue, "final")