import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, (3, 4), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, (3, 4), stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (3, 4), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_ffn = nn.Sequential(
            nn.Linear(10, 20)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        print(x.shape)
        return x