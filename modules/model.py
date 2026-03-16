import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_size, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.ffn(x)
