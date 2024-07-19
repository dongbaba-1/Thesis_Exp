import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import cv2
from collections import deque
import time


# DQN模型定义
class Qnet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 选择动作
def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()


# 优化模型
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.FloatTensor(batch[0])
    action_batch = torch.LongTensor(batch[1]).unsqueeze(1)
    reward_batch = torch.FloatTensor(batch[2])
    next_state_batch = torch.FloatTensor(batch[3])
    done_batch = torch.FloatTensor(batch[4])

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = F.mse_loss(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 从AirSim获取深度图
def get_depth_image(client):
    # 向AirSim请求深度图像
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)])
    response = responses[0]

    # 将深度图像数据转换为一维数组
    img1d = np.array(response.image_data_float, dtype=np.float32)

    # 将深度值归一化到0-255范围内
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)

    # 将一维数组转换为二维图像
    img2d = np.reshape(img1d, (response.height, response.width))

    # 将二维图像转换为灰度图像并取反
    image = np.invert(np.array(Image.fromarray(img2d).convert('L')))

    # 调整图像大小为84x84像素
    image = cv2.resize(image, (84, 84))

    # 将图像转换为浮点数组
    image = np.array(image, dtype=np.float32)

    return image


# 初始化AirSim客户端
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# DQN参数设置
state_dim = (1, 84, 84)
action_dim = 5  # 假设我们有5个动作：前进、后退、左转、右转、悬停
policy_net = Qnet(state_dim[0], action_dim)
target_net = Qnet(state_dim[0], action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = deque(maxlen=10000)

# 训练参数
num_episodes = 1000
batch_size = 64
gamma = 0.99
epsilon = 0.1
target_update = 10

# 训练循环
for episode in range(num_episodes):
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    state = get_depth_image(client)
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    total_reward = 0
    done = False
    step = 0

    while not done:
        action = select_action(state, policy_net, epsilon, action_dim)

        if action == 0:
            client.moveByVelocityAsync(1, 0, 0, 1).join()
        elif action == 1:
            client.moveByVelocityAsync(-1, 0, 0, 1).join()
        elif action == 2:
            client.moveByVelocityAsync(0, 1, 0, 1).join()
        elif action == 3:
            client.moveByVelocityAsync(0, -1, 0, 1).join()
        elif action == 4:
            client.hoverAsync().join()

        next_state = get_depth_image(client)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

        reward = -1  # 默认奖励
        if client.getMultirotorState().landed_state == airsim.LandedState.Landed:
            reward = -100
            done = True
        else:
            reward = 1
            done = False

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        step += 1

        optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f'Episode {episode}, Total Reward: {total_reward}')

client.armDisarm(False)
client.enableApiControl(False)
