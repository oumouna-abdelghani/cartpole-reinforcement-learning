
"""Train, evaluate, and save a DQN agent for CartPole."""

from pathlib import Path

import gym
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from model import build_model
from agent import build_agent


# Create the output directory if it doesn't exist.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Create the environment.
env = gym.make("CartPole-v0")

states = env.observation_space.shape[0]
actions = env.action_space.n

print(f"Number of state features: {states}")
print(f"Number of possible actions: {actions}")

# 2. Build the neural network.
model = build_model(states, actions)
model.summary()

# 3. Build and configure the DQN agent.
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])

# 4. Train the agent
dqn.fit(
    env,
    nb_steps=50_000,
    visualize=False,
    verbose=1
)

# Save the trained weights
dqn.save_weights("dqn_weights.h5f", overwrite=True)

# 5. Evaluate and visualize the trained agent
scores = dqn.test(
    env,
    nb_episodes=5,
    visualize=True,
    verbose=1
)

episode_rewards = scores.history["episode_reward"]
mean_reward = sum(episode_rewards) / len(episode_rewards)

print(f"\nMean evaluation reward: {mean_reward:.2f}")

# 6. Plot the evaluation rewards.
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.axhline(mean_reward, linestyle="--",
            label=f"Mean reward: {mean_reward:.1f}")

plt.title("DQN Agent - CartPole Evaluation")
plt.xlabel("Evaluation episode")
plt.ylabel("Episode reward")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "evaluation_rewards.png",
    dpi=150
)
plt.show()

env.close()
