from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:   # 经验回放池
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
    
class QNet(nn.Module): # Q网络
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN:
    def __init__(self,state_dim,action_dim,lr,buffer_capacity) -> None:
        
        self.eval_net = QNet(state_dim, action_dim).to(device) # 创建eval_net
        self.target_net = QNet(state_dim, action_dim).to(device) # 创建target_net
        self.target_net.load_state_dict(self.eval_net.state_dict()) # 初始化target_net的参数
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.steps = 0
        self.epsilon = 1.0

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon): # epsilon-greedy策略
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return self.predict(state)
        
    def predict(self, state): # 预测动作
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        return self.eval_net(state).argmax(dim=1).item()
        # argmax(dim=1) 按照第一个维度（dim=1）上取最大值的索引
        # 索引很关键，因为action_dim=2，所以索引为0或1，分别对应左右两个动作
    
    def update(self, batch_size, gamma, epsilon_decay): # 训练
        if len(self.buffer) < batch_size: # 经验池中的样本数量不足时，不进行训练
            return
        state, action, reward, next_state, done = self.buffer.sample(batch_size) # 从经验池中采样
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).view(-1, 1).to(device)
        # view(-1, 1) 将action的维度从(batch_size,)转换为(batch_size, 1)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(device)

        q_eval = self.eval_net(state).gather(1, action) # 从eval_net中获取Q(s,a)的值 
        # gather() 从第一个维度（dim=1）上选择action对应的值
        q_next = self.target_net(next_state).detach() # 从target_net中获取Q(s',a')的值
        # print(q_next)
        # detach() 从计算图中分离，不进行反向传播
        q_target = reward + gamma * q_next.max(dim=1, keepdim=True)[0] * (1 - done) # 计算Q(s,a)的目标值
        # max(dim=1, keepdim=True) 按照第一个维度（dim=1）上取最大值，keepdim=True 保持维度不变
        # 原因：q_next的维度为(batch_size, action_dim)，而reward和done的维度为(batch_size, 1) 
        # 因此需要在dim=1上取最大值，且需要保持维度不变
        loss = self.loss(q_eval, q_target) # 计算损失函数
        
        # 反向传播
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()

        # 更新epsilon
        self.steps += 1
        if self.steps % epsilon_decay == 0:
            self.epsilon = max(0.1, self.epsilon - 0.01)

        # 更新target_net的参数
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

def train(env, agent, episodes, batch_size, gamma, epsilon_decay): # 训练
    for i_episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state, agent.epsilon)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size, gamma, epsilon_decay)
            total_reward += reward
            state = next_state
            if done:
                break
        if i_episode % 10 == 0:
            print(f'Episode {i_episode} Reward {total_reward}')

def test(env, agent, episodes): # 测试
    for i_episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        print(f'Episode {i_episode + 1} Reward {total_reward}')

# env = gym.make('CartPole-v1')
# print(env.observation_space)             # Box(4,) 位置，速度，角度，角速度
# print(env.observation_space.shape[0])    # 4
# print(env.action_space.n)                # 2 0向左，1向右

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0] # 状态空间维度
action_dim = env.action_space.n # 动作空间维度
agent = DQN(state_dim, action_dim, lr=0.001, buffer_capacity=10000) 
train(env, agent, episodes=350, batch_size=64, gamma=0.99, epsilon_decay=200)
test(env, agent, episodes=10)
