# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
import torch.optim as optim
from torchsummary import summary


hyperparams = {
    'conv1_output': 4,
    'conv1_kernel': 3,
    'conv2_output': 8,
    'conv2_kernel': 5,
    'conv3_output': 16,
    'conv3_kernel': 7,
    'conv4_output': 4,
    'conv4_kernel': 7,
    'hidden1': 32,
    'hidden2': 64,
    'lr': 0.005,
    'gamma': 0.8
}


class QNetConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = f.relu(x)
        return x


class QNet(nn.Module):

    def __init__(self, grid_h, grid_w):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.conv1 = QNetConv2d(1, hyperparams['conv1_output'], hyperparams['conv1_kernel'])
        self.conv2 = QNetConv2d(hyperparams['conv1_output'], hyperparams['conv2_output'], hyperparams['conv2_kernel'])
        self.conv3 = QNetConv2d(hyperparams['conv2_output'], hyperparams['conv3_output'], hyperparams['conv3_kernel'])
        self.conv4 = QNetConv2d(hyperparams['conv3_output'], hyperparams['conv4_output'], hyperparams['conv4_kernel'])
        self.flatten = nn.Flatten()
        shrink = (hyperparams['conv1_kernel'] + hyperparams['conv2_kernel'] +
                  hyperparams['conv3_kernel'] + hyperparams['conv4_kernel'] - 4)
        self.linear1 = nn.Linear(
            hyperparams['conv4_output']*(grid_h-shrink)*(grid_w-shrink)+11,
            hyperparams['hidden1']
        )
        self.linear2 = nn.Linear(hyperparams['hidden1'], hyperparams['hidden2'])
        self.linear3 = nn.Linear(hyperparams['hidden2'], 3)

    def forward(self, x):
        direction_tensor = x[:, :11]
        obstacles_tensor = x[:, 11:]
        obstacles_tensor = obstacles_tensor.view(-1, 1, self.grid_h, self.grid_w)
        obstacles_tensor = self.conv1(obstacles_tensor)
        obstacles_tensor = self.conv2(obstacles_tensor)
        obstacles_tensor = self.conv3(obstacles_tensor)
        obstacles_tensor = self.conv4(obstacles_tensor)
        obstacles_tensor = self.flatten(obstacles_tensor)
        x = torch.concat([direction_tensor, obstacles_tensor], dim=1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = f.softmax(self.linear3(x), dim=1)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class ModelSimple(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.softmax(self.linear2(x), dim=1)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:

    def __init__(self, model):
        self.gamma = hyperparams['gamma']
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=hyperparams['lr'])
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        reward = torch.tensor(reward, dtype=torch.float)

        if state.shape[0] == 1:
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(
                    next_state[idx].view(-1, self.model.grid_h*self.model.grid_w+11))
                )
                # q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].view(-1, 11)))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
