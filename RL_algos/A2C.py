import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal, Categorical
import numpy as np
import matplotlib.pyplot as plt

lr_actor = 0.0001
lr_critic = 0.001

gamma = 0.95

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def length(self):
        return len(self.actions)
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Output mu, std
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.std = nn.Linear(64, action_dim)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu(x))
        std = F.softplus(self.std(x))

        return mu, std
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v
    
class A2C:
    def __init__(self, std_bound, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic = Critic(self.state_dim)
        self.buffer = RolloutBuffer()
        self.std_bound = std_bound
        self.BATCH_SIZE = 32
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr= lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.env = env

        self.reward_sum_lst = []

    def take_action(self, state):
        mu, std = self.actor(state)
        mu = mu * self.action_bound
        std_bound = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])

        dist = Normal(mu, std_bound)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def actor_learn(self, logprobs, advantages):
        self.actor_optimizer.zero_grad()
        advantages = advantages.squeeze().detach()

        loss = -logprobs * advantages
        loss = loss.mean()
        loss.backward()
        self.actor_optimizer.step()

    def critic_learn(self, states, td_targets):
        self.critic_optimizer.zero_grad()
        td_hat = self.critic(states)
        
        loss = F.smooth_l1_loss(td_hat.squeeze(), torch.FloatTensor(td_targets).detach()).mean()
        loss.backward()
        self.critic_optimizer.step()
        
    
    def td_target(self, rewards, next_v_values, dones):
        td_targets = np.zeros(len(next_v_values))
        for i in range(len(next_v_values)):
            if dones[i] == True:
                td_targets[i] = rewards[i]
            else:
                td_targets[i] = rewards[i] + gamma * next_v_values[i]

        return td_targets


    def train(self, max_episode_num):
        score = 0.0
        timestep = 0
        for epi in range(max_episode_num):
            episode_reward = 0
            state = self.env.reset()[0]
            done = False

            while not done:
                action, log_prob = self.take_action(torch.tensor(state).float())
                next_state, reward, terminated, truncated, info = self.env.step([action.item()])
                done = terminated or truncated

                self.buffer.states.append(torch.FloatTensor(state))
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(log_prob)
                self.buffer.rewards.append(torch.FloatTensor([reward/100.0]))
                self.buffer.next_states.append(torch.FloatTensor(next_state))

                done_mask = 1.0 if done else 0.0
                self.buffer.dones.append(torch.FloatTensor([done_mask]))

                if self.buffer.length() < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward
                    score += reward
                    timestep += 1
                    continue

                states = torch.squeeze(torch.stack(self.buffer.states, dim=0))
                log_probs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
                train_rewards = torch.squeeze(torch.stack(self.buffer.rewards, dim=0))
                next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0))
                dones = torch.squeeze(torch.stack(self.buffer.dones, dim=0))

                self.buffer.clear()
                next_v_values = self.critic(next_states)
                td_targets = self.td_target(train_rewards, next_v_values, dones)
                advantages = train_rewards.unsqueeze(dim=1) + gamma * next_v_values - self.critic(states)

                self.critic_learn(states, td_targets)
                self.actor_learn(log_probs, advantages)

                state = next_state
                score += reward
                timestep += 1

            if epi % 20 == 0 and epi != 0:
                print(f"# of episode: {epi}, timestep: {timestep/20}, score: {score/20}")
                self.reward_sum_lst.append(score/20)
                score = 0.0
                timestep = 0

    def draw_reward_sum(self):
        idx_len = len(self.reward_sum_lst)
        idx = [i*20 for i in range(idx_len)]

        plt.plot(idx, self.reward_sum_lst)
        plt.xlabel("# of Episode", fontsize=20)
        plt.ylabel("Reward Sum", fontsize=15)
        plt.savefig("A2C.png")



agent = A2C(
    std_bound=[1e-2, 1], 
    env=gym.make("Pendulum-v1")
)

agent.train(2500)
agent.draw_reward_sum()
