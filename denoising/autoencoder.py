import torch
import torch.nn as nn

# Hyperparameters
## CNN

## LSTM


# CNN

class CnnEncoder(nn.Module):
    def __init__(self, 
                 conv1_kernel=3, 
                 conv2_kernel=3, 
                 conv3_kernel=3, 
                 encoded_space_dim=0
                 ):
        super(CnnEncoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, conv1_kernel, stride=1, padding=conv1_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, conv2_kernel, stride=1, padding=conv2_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, conv3_kernel, stride=1, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, conv3_kernel, stride=1, padding=conv3_kernel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return x

class CnnDecoder(nn.Module):
    def __init__(self, 
                 conv1_kernel=3, 
                 conv2_kernel=3, 
                 conv3_kernel=3, 
                 encoded_space_dim=0
                 ):
        super(CnnDecoder, self).__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 32, conv3_kernel, stride=1, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, conv3_kernel, stride=1, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, conv2_kernel, stride=1, padding=conv2_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 1, conv1_kernel, stride=1, padding=conv1_kernel),
        )

    def forward(self, x):
        x = self.decoder_cnn(x)
        return x

# LSTM

class LstmEncoder(nn.Module):
    def __init__(self, encoded_space_dim) -> None:
        super(LstmEncoder).__init__()

        self.encoder_lstm = nn.LSTM(input_dim=1, )