#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 训练配置
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
DATA_PATH = os.path.expanduser("~/bc_data/tello_expert_data.pkl")
SAVE_PATH = os.path.expanduser("~/bc_data/tello_bc_policy.pth")

# 数据集定义
class ExpertDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, action, vel, alt = self.data[idx]
        image = self.transform(image)
        state = torch.tensor(vel + [alt], dtype=torch.float32)  # [vx, vy, vz, alt]
        action = torch.tensor(action, dtype=torch.float32)
        return image, state, action

# 模型定义
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
        )
        # 动态确定 flatten 后的尺寸
        dummy = torch.zeros(1, 3, 120, 160)
        conv_out = self.conv(dummy).view(1, -1)
        self.conv_out_dim = conv_out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_dim + 4, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, img, state):
        img_feat = self.conv(img).view(img.size(0), -1)
        x = torch.cat([img_feat, state], dim=1)
        return self.fc(x)


# 训练主函数
def train():
    dataset = ExpertDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for img, state, act in dataloader:
            pred = model(img, state)
            loss = loss_fn(pred, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 模型已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    train()

