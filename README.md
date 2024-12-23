# MAgent2 RL Final Project
## Overview
In this final project, we will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment `battle`, and our trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent, which will be released in the final week of the course before the deadline.

Our agent's performance will be evaluated based on reward and win rate against each of these models. We control "blue" agent while evaluating.


<p align="center">
  <img src="assets/random demo.gif" width="300" alt="random agent" />
  <br>
  <strong>Against Random Agent</strong>
</p>

<p align="center">
  <img src="assets/pretrained demo.gif" width="300" alt="pretrained agent" />
  <br>
  <strong>Against Pretrained Agent</strong>
</p>

<p align="center">
  <img src="assets/final demo.gif" width="300" alt="final agent" />
  <br>
  <strong>Against Final Agent</strong>
</p>

To recreate these demo see `eval/main.py` which load provided pre-trained policies using `pretrained_agent.py` containing wrappers for each of the agents.

Our reported agent was trained using Deep Q-Network with Double Q and Prioritized Replay Buffer using PyTorch, the implementation can be found at `solution/DeepQNetwork/Double Q`. This training script was executed on Kaggle with Accelerator setting of GPU T4 x2.

## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Evaluation
Refer to `eval/eval.py` for the evaluation code, this code also loads wrappers of our trained agents from `self_trained_agent.py`.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.
