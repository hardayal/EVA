"""

Automatically generated by Colaboratory.


# Twin-Delayed DDPG

On a custom car env


TODO: add swiggy food delivery system
"""

# #env render is made using pygame
# !pip install pygame

"""## Importing the libraries"""
import pygame
import gym_swiggyfood
from gym import wrappers
import gym
from PIL import Image as PILImage
import math
from collections import deque
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import sys




"""## Step 1: We initialize the Experience Replay memory"""

class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
        for i in ind:
            state1, state2, next_state1, next_state2, action, reward, done = self.storage[i]
            batch_states1.append(state1)
            batch_states2.append(np.array(state2, copy=False))
            batch_next_states1.append(next_state1)
            batch_next_states2.append(np.array(next_state2, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states1), np.array(batch_states2), np.array(batch_next_states1), np.array(batch_next_states2), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


"""## Step 2: We build one neural network for the Actor model and one neural network for the Actor target"""


# def conv2d_size_out(size, kernel_size=3, stride=2):
#     return (size - (kernel_size - 1) - 1) // stride + 1
# conv2d_size_out(conv2d_size_out(60))


class AC_conv(nn.Module):

    def __init__(self, state_dim=1):
        super(AC_conv, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)  # sq of an odd number, because just!
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # print(x.shape)
        return torch.nn.functional.avg_pool2d(x, kernel_size=5, stride=5)
        # return F.relu(self.conv3(x))


# Actor Models
class Actor(AC_conv):
    def __init__(self, state_dim, action_dim, max_action):
        AC_conv.__init__(self)
        super(Actor, self).__init__()

        linear_input_size = 25+3
        self.layer_1 = nn.Linear(linear_input_size, 30)  # if on road or sand
        self.layer_2 = nn.Linear(30, 50)
        self.layer_3 = nn.Linear(50, action_dim)

        self.max_action = max_action

    def forward(self, x1, x2):
        x1 = AC_conv.forward(self, x1)

        x = torch.cat(((x1.view(x1.size(0), -1)),
                       x2), 1)

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))


"""## Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets"""


# new
class Critic(AC_conv):

    def __init__(self, state_dim, action_dim):
        AC_conv.__init__(self)
        super(Critic, self).__init__()
        # Defining the first Critic neural network

        linear_input_size = 25+3  # add state["orientation"]

        self.layer_1 = nn.Linear(linear_input_size + action_dim, 30)  # if on road or sand
        self.layer_2 = nn.Linear(30, 50)
        self.layer_3 = nn.Linear(50, 1)

        # Defining the second Critic neural network

        self.layer_4 = nn.Linear(linear_input_size + action_dim, 30)  # if on road or sand
        self.layer_5 = nn.Linear(30, 50)
        self.layer_6 = nn.Linear(50, 1)

    def forward(self, x1, x2, u):
        # Forward-Propagation on the first Critic Neural Network
        x1_1 = AC_conv.forward(self, x1)

        xu_1 = torch.cat(((x1_1.view(x1_1.size(0), -1)),
                          x2, u), 1)

        x_1 = F.relu(self.layer_1(xu_1))
        x_1 = F.relu(self.layer_2(x_1))
        x_1 = self.layer_3(x_1)

        # Forward-Propagation on the second Critic Neural Network
        # x1_2 = self.mp2(x1)

        x1_2 = AC_conv.forward(self, x1)

        xu_2 = torch.cat(((x1_2.view(x1_1.size(0), -1)),
                          x2, u), 1)

        x_2 = F.relu(self.layer_4(xu_2))
        x_2 = F.relu(self.layer_5(x_2))
        x_2 = self.layer_6(x_2)
        return x_1, x_2

    def Q1(self, x1, x2, u):
        x1_1 = AC_conv.forward(self, x1)

        xu_1 = torch.cat(((x1_1.view(x1_1.size(0), -1)),
                          x2, u), 1)

        x_1 = F.relu(self.layer_1(xu_1))
        x_1 = F.relu(self.layer_2(x_1))
        x_1 = self.layer_3(x_1)
        return x_1


"""## Steps 4 to 15: Training Process"""

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state1, state2):
        state1 = torch.from_numpy(state1).float().permute(
            2, 0, 1).unsqueeze(0).to(device)
        state2 = torch.Tensor(state2).unsqueeze(0).to(device)
        # print(f'shape of state1: {state1.shape}; state2{state2.shape}')
        return self.actor(state1, state2).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state1 = torch.from_numpy(batch_states1).float().permute(0, 3, 1, 2).to(device)
            state2 = torch.Tensor(batch_states2).to(device)
            # next_state1 = torch.Tensor(batch_next_states1).to(device)
            next_state1 = torch.from_numpy(batch_next_states1).float().permute(0, 3, 1, 2).to(device)
            next_state2 = torch.Tensor(batch_next_states2).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state1, next_state2)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state1, next_state2, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state1, state2, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state1, state2,
                                             self.actor(state1, state2)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(
            '%s/%s_actor.pth' % (directory, filename), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(
            '%s/%s_critic.pth' % (directory, filename), map_location=lambda storage, loc: storage))


"""## We make a function that evaluates the policy by calculating its average reward over 10 episodes"""


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        # print(f'pickup{env.x1, env.y1}; drop{env.x2,env.y2}')
        done = False
        while not done:
            action = policy.select_action(obs['surround'], obs['orientation'])
            obs, reward, done, _ = env.step(action)
            env.render()

            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """## We set the parameters"""

    env_name = "SwiggyFood-v0"
    # seed = 0 # Random seed number
    start_timesteps = 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5  # Total number of iterations/timesteps
    save_models = True  # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100  # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005  # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5# Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2# Number of iterations to wait before the policy network (Actor model) is updated

    """## We create a file name for the two saved models: the Actor and Critic models"""

    file_name = "%s_%s" % ("TD3", env_name)

    """## We create a folder inside which will be saved the trained models"""
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    """## We create the PyBullet environment"""
    env = gym.make(env_name)
 
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    state_dim = env.observation_space["surround"].shape[2]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])


    """ ## We create the policy network (the Actor model)"""
    policy = TD3(state_dim, action_dim, max_action)

    """## We create the Experience Replay memory"""
    replay_buffer = ReplayBuffer()

    """## We define a list where all the evaluation results over 10 episodes are stored"""
    evaluations = [evaluate_policy(policy)]

    max_episode_steps = env._max_episode_steps

    """## We initialize the variables"""

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()

    """## Training"""

    max_timesteps = 500000
    # We start the main loop over 500,000 timesteps
    while total_timesteps < max_timesteps:
        # print(f'timestamp: {abc}')
        env.render()

        # If the episode is done
        if done:

            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {} on road: {} off road: {}".format(total_timesteps, episode_num, episode_reward, on_road, off_road))
                policy.train(replay_buffer, episode_timesteps, batch_size,
                            discount, tau, policy_noise, noise_clip, policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(policy))
                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # When the training step is done, we reset the state of the environment
            obs = env.reset()

            # Set the Done to False
            done = False

            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            #pos and neg reward counter
            on_road = 0
            off_road = 0

        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else:  # After 10000 timesteps, we switch to the model
            # action = policy.select_action(np.array(obs))
            action = policy.select_action(obs['surround'], obs['orientation'])

            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)

        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        # We increase the total reward
        episode_reward += reward

          # see pos and neg reward counts
        if reward >= 0.0:
            on_road += 1
        else:
            off_road += 1  

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs['surround'], obs['orientation'], new_obs['surround'],
                        new_obs['orientation'], action, reward, done_bool))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy))
    if save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
    env.close()
