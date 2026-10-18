# CartPole Reinforcement Learning

My first hands-on reinforcement learning project after completing the University of Alberta's *Fundamentals of Reinforcement Learning* course.

The project trains a Deep Q-Network (DQN) agent to balance a pole on a moving cart using TensorFlow/Keras, `keras-rl2`, and OpenAI Gym.

## Run locally

Requires Python 3.8 and the dependencies listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python main.py
```

The script trains and evaluates the agent, saves its weights, and generates an evaluation-reward plot in `results/`.

## Tech stack

- Python
- TensorFlow / Keras
- keras-rl2 (DQN)
- OpenAI Gym

## Future improvement

Implement DQN from scratch using PyTorch to better understand its core components, including experience replay, target networks, and Bellman updates.