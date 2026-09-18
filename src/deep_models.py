"""
1D CNN and LSTM models for raw-signal human activity recognition,
implemented in PyTorch. Both take a window of shape (window_size, 6)
(6 = back_x/y/z, thigh_x/y/z) and predict one of n_classes activities.

Kept deliberately compact: this project's compute budget is a single CPU
core, so the models favor being fast and reasonably capable over being
state-of-the-art large.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 12, window_size: int = 150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool -> works for any window_size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, window_size, n_channels) -> conv1d wants (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 12, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, window_size, n_channels)
        out, (h_n, c_n) = self.lstm(x)
        # concat final forward + backward hidden states
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(last)
