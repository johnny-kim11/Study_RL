import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

lr_actor = 0.00005
lr_critic = 0.0005

gamma = 0.99

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
    
class PPO:
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
        self.GAE_LAMBDA = 0.9
        self.RATIO_CLIPPING = 0.05
        self.EPOCHs = 10

        self.reward_sum_lst = []

    def take_action(self, state):
        mu, std = self.actor(state)
        mu = mu * self.action_bound
        std_bound = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])
        cov_mat = torch.diag(std_bound*std_bound).unsqueeze(dim=0)

        dist = MultivariateNormal(mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def get_new_data(self, states, actions):
        dist_entropy_lst = []
        log_probs_lst = []

        # state, action 입력받으면 log_prob, V(s), dist_entropy 출력 
        mu, std = self.actor(states)
        std_bound = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])
        for i in range(len(states)):
            mu, std = self.actor(states[i])
            std = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])
            cov_mat = torch.diag(torch.square(std)).unsqueeze(dim=0)
            dist = MultivariateNormal(mu, cov_mat)
            log_probs = dist.log_prob(actions[i])
            dist_entropy = dist.entropy()

            dist_entropy_lst.append(dist_entropy)
            log_probs_lst.append(log_probs)

        dist_entropy = torch.squeeze(torch.stack(dist_entropy_lst, dim=0))
        log_probs = torch.squeeze(torch.stack(log_probs_lst, dim=0))
        
        return log_probs, dist_entropy
    
    def gaes(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0
        v_values = v_values.squeeze()

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + gamma*forward_val - v_values[k]
            gae_cumulative = gamma * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]

        return gae, n_step_targets
    
    def actor_learn(self, old_logprobs, states, actions, gaes):
        self.actor_optimizer.zero_grad()

        gaes = torch.FloatTensor(gaes)
        
        log_probs, dist_entropy = self.get_new_data(states, actions)
        ratios = torch.exp(log_probs - old_logprobs.detach())

        surr1 = ratios * gaes
        surr2 = torch.clamp(ratios, 1-self.RATIO_CLIPPING, 1+self.RATIO_CLIPPING) * gaes
        surrogate = torch.min(surr1, surr2)

        loss = -surrogate - dist_entropy*0.01
        loss = loss.mean()
        loss.backward()
        self.actor_optimizer.step()

    def critic_learn(self, states, td_targets):
        self.critic_optimizer.zero_grad()
        td_hat = self.critic(states)
        
        loss = F.smooth_l1_loss(td_hat.squeeze(), torch.FloatTensor(td_targets).detach()).mean()
        loss.backward()
        self.critic_optimizer.step()



    def train(self, max_episode_num):
        score = 0.0
        timestep = 0
        for epi in range(max_episode_num):
            episode_reward = 0
            state = self.env.reset()[0]
            done = False

            while not done:
                action, log_prob = self.take_action(torch.tensor(state).float())
                next_state, reward, terminated, truncated, info = self.env.step(action.numpy()[0])
                done = terminated or truncated

                self.buffer.states.append(torch.FloatTensor(state))
                self.buffer.actions.append(torch.FloatTensor(action))
                self.buffer.logprobs.append(log_prob)
                self.buffer.rewards.append(torch.FloatTensor([reward/10.0]))
                self.buffer.next_states.append(torch.FloatTensor(next_state))

                done_mask = 1.0 if done else 0.0
                self.buffer.dones.append(torch.FloatTensor([done_mask]))

                if self.buffer.length() < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward
                    score += reward
                    timestep += 1
                    continue

                old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0))
                old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0))
                old_log_probs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
                old_train_rewards = torch.squeeze(torch.stack(self.buffer.rewards, dim=0))

                self.buffer.clear()
                next_v_value = self.critic(torch.FloatTensor(next_state))
                v_values = self.critic(old_states)
                gaes, n_td_targets = self.gaes(old_train_rewards, v_values, next_v_value, done_mask)

                for _ in range(self.EPOCHs):
                    self.critic_learn(old_states, n_td_targets)
                    self.actor_learn(old_log_probs, old_states, old_actions, gaes)

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
        plt.xlabel("# of Episode", fontsize=15)
        plt.ylabel("Reward Sum", fontsize=15)
        plt.tight_layout()
        plt.savefig("PPO.png")



agent = PPO(
    std_bound=[1e-2, 1], 
    env=gym.make("LunarLanderContinuous-v2")
)

agent.train(3000)
agent.draw_reward_sum()
