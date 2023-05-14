import gym
import numpy as np
import torch
import random
import torch.nn as nn
import copy
from matplotlib import pyplot as plt

from gym.wrappers import RecordVideo

LEARNING_RATE = 0.0005
GAMMA = 0.95
ENV_NAME = 'LunarLander-v2' #'CartPole-v1' # 
EPSILON = 0.1


# Q function Network 
class agent(nn.Module):
    def __init__(self) -> None:
        super(agent, self).__init__()
        self.env = gym.make(ENV_NAME, continuous=False)
        # self.env.seed(0)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fcq = nn.Linear(128, self.output_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q = self.fcq(x)
        return q
    
# Training(Q-Learning) Q function Network
''' 
    1. Calculate Q(s_t,a_t) by using Q function Network.
    2. Sample the action by using epsilon-greedy policy.
    3. Obtain next state, reward
    4. Calculate Q(s_t+1, a_t+1) by using Q function Network.
    5. Calculate loss by using MSE loss function.
       Loss = (Q(s_t,a_t) - (r_t+1 + gamma * maxQ(s_t+1, a_t+1)))^2
    6. Update Q function Network by using Adam optimizer.
    7. Decrease epsilon since the Q function would be accurate as the number of training step increases.
    8. Save the model if you think the model is good enough.
'''
class Q_learning():
    def __init__(self):
        self.agent = agent()
        self.env = gym.make(ENV_NAME, continuous=False)
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=LEARNING_RATE)

    def train(self):
        print('------------Start Training--------------')
        reward_list = []
        max_reward = 0
        for epoch in range(10000):
            done = False
            state_ = self.env.reset()[0]
            reward_sum = 0
            step = 0
            while(not done):
                # (1)
                state = torch.from_numpy(state_).float()
                q_val = self.agent(state)
                
                # (2)
                if (random.random() < self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_val).item()
                    
                # (3)
                next_state_, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state_).float()
                step += 1
                done = terminated or truncated
                
                # (4)
                reward_sum += reward
                newQ = self.agent(next_state).detach()
                maxQ = torch.max(newQ)
                
                # (5)
                X = q_val[action].unsqueeze(0)

                if done:
                    Y = reward
                else:
                    Y = reward + self.gamma * maxQ
                Y = torch.Tensor([Y]).detach()
                loss = self.loss_fn(X, Y)
                
                # (6)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # (7)
            # if self.epsilon > 0.05:
            #     self.epsilon -= 1/(1+epoch)
                
                state_ = next_state_

            print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))
            reward_list.append(reward_sum)
            
            if reward_sum >= max_reward:
                max_reward = reward_sum
                torch.save(copy.deepcopy(self.agent.state_dict()), "Q-learning.pt")
                print("Model Saved")
            
            # (8)
            # if reward_sum > 200 and epoch > 2500:
            #     torch.save(copy.deepcopy(self.agent.state_dict()), "Q-learning.pt")
            #     break
            
        self.env.close()
        
        n_epochs = len(reward_list)
        plt.plot(np.arange(n_epochs), reward_list)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.savefig('Q-learning_reward.png')


def test(model):
    print('------------Start Test--------------')
    model.eval()
    env = gym.make(ENV_NAME, continuous=False)
    for epoch in range(10):
        done = False
        state_ = env.reset()[0]
        reward_sum = 0
        step = 0
        while(not done):
            state = torch.from_numpy(state_).float()
            q_val = model(state)
            action = torch.argmax(q_val).item()
            
            next_state_, reward, terminated, truncated, _ = env.step(action)       
            reward_sum += reward
            state_ = next_state_
            done = terminated or truncated
            
            step += 1
        print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))
            
            
def test_video(model):
    print('------------Start Test--------------')
    model.eval()
    env = RecordVideo(gym.make(game=ENV_NAME, continuous=False), "./")
    for epoch in range(1):
        done = False
        state_ = env.reset()[0]
        reward_sum = 0
        step = 0
        while(not done):
            state = torch.from_numpy(state_).float()
            q_val = model(state)
            action = torch.argmax(q_val).item()
            
            next_state_, reward, terminated, truncated, _ = env.step(action)     
            reward_sum += reward
            state_ = next_state_
            done = terminated or truncated
            
            step += 1
        print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))
if __name__ == "__main__":
    model = Q_learning()
    # model.train()

    test_model = agent()
    test_model.load_state_dict(torch.load("Q-learning.pt"))
    test_video(test_model)