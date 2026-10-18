
"""Define the neural network used by the DQN agent."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def build_model(states, actions):
    model = Sequential()

    # The agent supplies one state containing four observations.
    model.add(Flatten(input_shape=(1, states)))

    # Two hidden layers learn relationships between observations.
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))

    # One Q-value for each possible action.
    model.add(Dense(actions, activation="linear"))

    return model
