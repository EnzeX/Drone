#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import random

# Training configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DATA_PATH = os.path.expanduser("~/bc_data/airsim_expert_data_.pkl")
SAVE_PATH = os.path.expanduser("~/bc_data/airsim_bc_policy_vit.pth")

def set_seed(seed=42):
    random.seed(seed)             
    np.random.seed(seed)           
    torch.manual_seed(seed)        
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# dataset
class ExpertDataset(Dataset):
    def __init__(self, pkl_path):
        expanded_path = os.path.expanduser(pkl_path)
        with open(expanded_path, "rb") as f:
            self.data = pickle.load(f)

        weights = ViT_B_16_Weights.DEFAULT
        # mean, std = weights.meta["mean"], weights.meta["std"]
        self.preprocess = weights.transforms()
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        
        self.max_vel = 300 
        self.max_alt = 1600

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, action, vel, alt = self.data[idx]
        image = to_pil_image(image)
        image = self.preprocess(image)
        
        norm_vx = np.clip(vel[0] / self.max_vel, -1.0, 1.0)
        norm_vy = np.clip(vel[1] / self.max_vel, -1.0, 1.0)
        norm_vz = np.clip(vel[2] / self.max_vel, -1.0, 1.0)
        norm_alt = np.clip(alt / self.max_alt, 0.0, 1.0)
        
        state = torch.tensor([norm_vx, norm_vy, norm_vz, norm_alt], dtype=torch.float32)  # [vx, vy, vz, alt]
        action = torch.tensor(action, dtype=torch.float32)
        return image, state, action


class PolicyNet(nn.Module):
    def __init__(self, freeze_vit=True):
        super().__init__()

        # Pretrained ViT backbone
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)

        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # output will be [B, vit_dim]

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False


        
        # compress image feature (optional bottleneck like your old design)
        self.image_bottleneck = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + 32, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, img, state):
        img_feat = self.vit(img)     # [B, vit_dim]
        img_compressed = self.image_bottleneck(img_feat)
        state_encoded = self.state_encoder(state)
        x = torch.cat([img_compressed, state_encoded], dim=1)
        return self.fc(x)


def train():
    set_seed(42)
    dataset = ExpertDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet(freeze_vit=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        total_samples = 0

        for img, state, act in dataloader:
            img = img.to(device)
            state = state.to(device)
            act = act.to(device)
            pred = model(img, state)
            loss = loss_fn(pred, act)

            # accumulate for metrics
            err = (pred - act).detach().cpu()
            total_mae += torch.abs(err).sum().item()
            total_rmse += (err ** 2).sum().item()
            total_samples += act.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # compute averages
        avg_mse = total_loss / max(1, len(dataloader))
        avg_mae = total_mae / max(1, total_samples * 4)  # 4 outputs
        avg_rmse = (total_rmse / max(1, total_samples * 4)) ** 0.5

        print(f"Epoch {epoch+1}/{EPOCHS}, Train MSE: {avg_mse:.4f}, Train MAE: {avg_mae:.4f}, Train RMSE: {avg_rmse:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    set_seed(42)
    train()

