from __future__ import annotations

from collections import defaultdict  # allows access to undefined keys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # draw shapes
import numpy as np
import seaborn as sns
from tqdm import tqdm  # progress bar
import gymnasium as gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics


env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")

env.reset()

# observing the environment
# reset the environment to get first observation
# done = False
# observation, info = env.reset()

# # observations are a tuple consisting of 3 values
# # observation = (16,9,False)
# # The players current sum
# # Values of the dealers face-up card
# # boolean that says if there us a usable 11 (ace)

# # sample a random action
# action = env.action_space.sample()

# # information received after taking an action
# observation, reward, terminated, truncated, info = env.step(action)
# observation=(24,10,False)
# reward = -1.0 - rewards the agent (in this case it's negative because it lost)
# terminated=True - Used to kill the environment (use env.reset())
# truncated = false
# info={}


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """_summary_
        gym.openai.com
        Args:
            learning_rate (float): Influences how quickly the q values change
            initial_epsilon (float): initial value (most likely going to be something like 1)
            epsilon_decay (float): Rate of epsilon decay
            final_epsilon (float): Final epsilon state
            discount_factor (float, optional): used to compute the Q-value. Defaults to 0.95.
        """

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """_summary_
        Args:
            obs (tuple[int,int,bool]): Agents observation as described above

            Returns:
                int: a specific action (hit or stick : 0,1)
        """
        if np.random.random() < self.epsilon:
            # for less than epsilon make a random action
            return env.action_space.sample()

        else:
            # else return the max q value to make the choice
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        # updating the q values
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    # Note for future. This is a way to make the agent start with completely random actions and slowly reduce the randomness
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

from IPython.display import clear_output

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    clear_output()

    # play one episode:
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            next_obs=next_obs,
        )
        frame = env.render()
        plt.imshow(frame)
        plt.show()

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode Rewards")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode Lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    training_error_moving_average = (
        np.convolve(
            np.array(agent.training_error).flatten(),
            np.ones(rolling_length),
            mode="same",
        )
        / rolling_length
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
    plt.tight_layout()
    plt.show()
