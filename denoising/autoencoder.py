import torch
import torch.nn as nn

class CnnEncoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(CnnEncoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, 3, stride=1),
            #nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return x

class CnnDecoder(nn.Module):
    def __init__(self, encoded_space_dim) -> None:
        super(CnnDecoder, self).__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 1, 3, stride=1)
        )

    def forward(self, x):
        x = self.decoder_cnn(x)
        return x
        