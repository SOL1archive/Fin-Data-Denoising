import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, fc2):
        super().__init__()