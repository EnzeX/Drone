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
import random

# Training configuration
BATCH_SIZE = 16  # Smaller batch for LSTM sequences
EPOCHS = 20
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 8  # Number of timesteps to remember
DATA_PATH = os.path.expanduser("~/bc_data/airsim_expert_data_.pkl")
SAVE_PATH = os.path.expanduser("~/bc_data/airsim_bc_policy_lstm_cnn.pth")

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
        
        # Create sequences from the data
        # Each sequence is SEQUENCE_LENGTH consecutive observations
        self.sequences = []
        for i in range(len(raw_data) - sequence_length + 1):
            self.sequences.append(raw_data[i:i + sequence_length])
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])
        
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
            # Transform image
            img_tensor = self.transform(image)
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
        
        # Stack into tensors: [sequence_length, ...]
        images = torch.stack(images)      # [seq_len, 3, 120, 160]
        states = torch.stack(states)      # [seq_len, 4]
        actions = torch.stack(actions)    # [seq_len, 4]
        
        return images, states, actions


class LSTMPolicyNet(nn.Module):
    """Policy network with LSTM memory for temporal reasoning"""
    def __init__(self, hidden_size=128, num_lstm_layers=2):
        super().__init__()
        
        # CNN encoder for visual features (same as before)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Determine conv output dimension
        dummy = torch.zeros(1, 3, 120, 160)
        conv_out = self.conv(dummy).view(1, -1)
        self.conv_out_dim = conv_out.shape[1]
        
        # Compress image features
        self.image_encoder = nn.Sequential(
            nn.Linear(self.conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        # Input: concatenated image + state features (64 + 32 = 96)
        self.lstm = nn.LSTM(
            input_size=64 + 32,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

    def forward(self, images, states, hidden=None):
        """
        Args:
            images: [batch, seq_len, 3, 120, 160]
            states: [batch, seq_len, 4]
            hidden: (h, c) tuple for LSTM, or None
        Returns:
            actions: [batch, seq_len, 4]
            hidden: (h, c) tuple for next step
        """
        batch_size, seq_len = images.shape[0], images.shape[1]
        
        # Process each timestep through CNN
        # Flatten batch and sequence dimensions for CNN
        images_flat = images.view(batch_size * seq_len, 3, 120, 160)
        
        # Extract visual features
        conv_features = self.conv(images_flat).view(batch_size * seq_len, -1)
        image_features = self.image_encoder(conv_features)  # [batch*seq, 64]
        image_features = image_features.view(batch_size, seq_len, -1)  # [batch, seq, 64]
        
        # Process state features
        states_flat = states.view(batch_size * seq_len, -1)
        state_features = self.state_encoder(states_flat)  # [batch*seq, 32]
        state_features = state_features.view(batch_size, seq_len, -1)  # [batch, seq, 32]
        
        # Concatenate features
        combined = torch.cat([image_features, state_features], dim=-1)  # [batch, seq, 96]
        
        # Pass through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(combined)
        else:
            lstm_out, hidden = self.lstm(combined, hidden)
        
        # Decode actions for each timestep
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        actions_flat = self.action_head(lstm_out_flat)
        actions = actions_flat.view(batch_size, seq_len, 4)
        
        return actions, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


def train():
    set_seed(42)
    dataset = SequenceExpertDataset(DATA_PATH, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPolicyNet(hidden_size=128, num_lstm_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print(f"Training on {len(dataset)} sequences of length {SEQUENCE_LENGTH}")
    print(f"Device: {device}")

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
            
            # Compute loss over all timesteps
            loss = loss_fn(pred_actions, actions)
            
            # Metrics
            with torch.no_grad():
                err = (pred_actions - actions).cpu()
                total_mae += torch.abs(err).sum().item()
                total_samples += actions.numel()  # Total number of action values
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping for LSTM
            optimizer.step()
            
            total_loss += loss.item()

        # Compute averages
        avg_loss = total_loss / max(1, len(dataloader))
        avg_mae = total_mae / max(1, total_samples)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Train MAE: {avg_mae:.4f}")

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
