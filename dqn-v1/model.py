import os
import typing as typ

import numpy as np
import torch


class CNN_QNet(torch.nn.Module):
    def __init__(self, input_shape, n_classes):
        super(CNN_QNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(input_shape[0] * input_shape[1] * 64, 512)
        self.fc2 = torch.nn.Linear(512, n_classes)

    def forward(self, x):
        print('Forward:', x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state : np.ndarray, action : np.ndarray, reward : typ.Union[int, np.ndarray], next_state : np.ndarray, done : typ.Union[bool, np.ndarray]):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        # # (n, x)
        print('Preprocess:', state.shape)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        print('Postprocess:', state.shape)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()