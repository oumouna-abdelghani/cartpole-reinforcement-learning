
"""Configure the DQN agent using keras-rl2."""

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    # Stores previous experiences for replay during learning.
    memory = SequentialMemory(
        limit=50_000,
        window_length=1
    )

    # Chooses actions according to their estimated Q-values.
    policy = BoltzmannQPolicy()

    # Creates the DQN agent.
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2
    )

    return dqn
