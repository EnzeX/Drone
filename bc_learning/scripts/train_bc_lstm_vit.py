#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Training configuration
BATCH_SIZE = 8  # Smaller batch for ViT+LSTM (memory intensive)
EPOCHS = 20
LEARNING_RATE = 5e-5  # Lower LR for ViT
SEQUENCE_LENGTH = 8
DATA_PATH = os.path.expanduser("~/bc_data/airsim_expert_data_.pkl")
SAVE_PATH = os.path.expanduser("~/bc_data/airsim_bc_policy_lstm_vit.pth")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceExpertDataset(Dataset):
    """Dataset that returns sequences of observations for LSTM training"""
    def __init__(self, pkl_path, sequence_length=8):
        expanded_path = os.path.expanduser(pkl_path)
        with open(expanded_path, "rb") as f:
            raw_data = pickle.load(f)
        
        # Create sequences
        self.sequences = []
        for i in range(len(raw_data) - sequence_length + 1):
            self.sequences.append(raw_data[i:i + sequence_length])
        
        weights = ViT_B_16_Weights.DEFAULT
        self.preprocess = weights.transforms()
        
        self.max_vel = 300 
        self.max_alt = 1600
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        images = []
        states = []
        actions = []
        
        for image, action, vel, alt in sequence:
            # Transform image with ViT preprocessing
            pil_img = to_pil_image(image)
            img_tensor = self.preprocess(pil_img)
            images.append(img_tensor)
            
            # Normalize state
            norm_vx = np.clip(vel[0] / self.max_vel, -1.0, 1.0)
            norm_vy = np.clip(vel[1] / self.max_vel, -1.0, 1.0)
            norm_vz = np.clip(vel[2] / self.max_vel, -1.0, 1.0)
            norm_alt = np.clip(alt / self.max_alt, 0.0, 1.0)
            state = torch.tensor([norm_vx, norm_vy, norm_vz, norm_alt], dtype=torch.float32)
            states.append(state)
            
            # Action
            action_tensor = torch.tensor(action, dtype=torch.float32)
            actions.append(action_tensor)
        
        # Stack into tensors
        images = torch.stack(images)      # [seq_len, 3, 224, 224]
        states = torch.stack(states)      # [seq_len, 4]
        actions = torch.stack(actions)    # [seq_len, 4]
        
        return images, states, actions


class LSTMViTPolicyNet(nn.Module):
    """Policy network with ViT encoder + LSTM memory"""
    def __init__(self, hidden_size=128, num_lstm_layers=2, freeze_vit=True):
        super().__init__()
        
        # Pretrained ViT backbone
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # Remove classification head
        
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        
        # Compress ViT features
        self.image_encoder = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=128 + 32,  # image + state features
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)
        )
        
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

    def forward(self, images, states, hidden=None):
        """
        Args:
            images: [batch, seq_len, 3, 224, 224]
            states: [batch, seq_len, 4]
            hidden: (h, c) tuple for LSTM
        Returns:
            actions: [batch, seq_len, 4]
            hidden: updated (h, c) tuple
        """
        batch_size, seq_len = images.shape[0], images.shape[1]
        
        # Process images through ViT
        images_flat = images.view(batch_size * seq_len, 3, 224, 224)
        
        with torch.set_grad_enabled(self.training):
            vit_features = self.vit(images_flat)  # [batch*seq, vit_dim]
        
        image_features = self.image_encoder(vit_features)  # [batch*seq, 128]
        image_features = image_features.view(batch_size, seq_len, -1)
        
        # Process state features
        states_flat = states.view(batch_size * seq_len, -1)
        state_features = self.state_encoder(states_flat)
        state_features = state_features.view(batch_size, seq_len, -1)
        
        # Combine features
        combined = torch.cat([image_features, state_features], dim=-1)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(combined)
        else:
            lstm_out, hidden = self.lstm(combined, hidden)
        
        # Decode actions
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        actions_flat = self.action_head(lstm_out_flat)
        actions = actions_flat.view(batch_size, seq_len, 4)
        
        return actions, hidden
    
    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


def train():
    set_seed(42)
    dataset = SequenceExpertDataset(DATA_PATH, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMViTPolicyNet(
        hidden_size=128, 
        num_lstm_layers=2,
        freeze_vit=True  # Freeze ViT to avoid overfitting
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"Training ViT+LSTM on {len(dataset)} sequences")
    print(f"Device: {device}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        for images, states, actions in dataloader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            pred_actions, _ = model(images, states)
            loss = loss_fn(pred_actions, actions)
            
            # Metrics
            with torch.no_grad():
                err = (pred_actions - actions).cpu()
                total_mae += torch.abs(err).sum().item()
                total_samples += actions.numel()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        avg_mae = total_mae / max(1, total_samples)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'num_lstm_layers': model.num_lstm_layers,
        'sequence_length': SEQUENCE_LENGTH
    }, SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    set_seed(42)
    train()
