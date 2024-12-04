import gym

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

import matplotlib.pyplot as plt
from IPython import display

# Start the environment
state = env.reset()

img = plt.imshow(env.render())  # Only call this once

for _ in range(1000):
    img.set_data(env.render())  # Just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)

    # Take a random action
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)

    if done:
        state = env.reset()
