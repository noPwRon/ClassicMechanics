import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# is_ipython = "inline" in matplotlib.get_backend
# if is_ipython:
from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# reply memory allows one to store and manage experiences during training


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# implementing a Deep Q network
# network has one hidden layer 128 nodes deep (wide?) and 3 layers in total

# N_observations is impute params: (theta of the pendulum, x on the path)
# #n_actions is # of possible actions (in this case 2)

# The class below has been generalized and could handle different numbers of actions and observations


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # take and pass to layer 3 of the neural network
    # output is the action it will take
    def fowrard(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# number of transitions sampled from the replay buffer
batch_size = 128
# Gamma is the discount factor
gamma = 0.99
eps_start = 0.9
eps_end = 0.06
eps_decay = 1000  # higher values will lead to a slower decay
tau = 0.005  # update rate of the target network
lr = 1e-4  # learning rate of the AdamW optimizer (Research Adaptive Moment Estimation)


n_actions = env.action_space.n

state, info = env.reset()

n_observations = len(state)

# target net is initilized with the same weight as policy net
policy_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
target_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


# AdamW will optimize these weights
optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row
            # second column on max results is index of where max element was found
            # will then pick the largest action
            return policy_net(state).max(1)[1].view(1, 1)

    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


episode_durations = []


def plot_duration(show_result=False):
    plt.figure(1)
    duration_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeroes(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause so that plots are update
    display.display(plt.gcf())
    
def optimize_mode():
    if len(memory) < batch_size:
        return 
    transition = memory.sample(batch_size)
    # converts batch array of transitions to transition of batch arrays
    batch = Transition(*zip(*transition))
    
    non_final_mask = torch.tensor(tuple(map(lamba s: s is not None),batch.next_state, device=device,dtype=torch.bool))
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    actions_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1,actions_batch)
    
    next_state_values = torch.zeros(batch_size,device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    expected_state_action_values = (next_state_values * gamma)
    reward_batch
        
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))
    
    #optimizing
    optimizer.zero_grad()
    loss.backward()
    
    #in-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(commander))