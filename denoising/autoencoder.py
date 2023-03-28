import torch
import torch.nn as nn

#################################################################################################
# CNN
#################################################################################################

class CnnEncoder(nn.Module):
    def __init__(self, 
                 conv1_kernel=3, 
                 conv2_kernel=3, 
                 conv3_kernel=3, 
                 stride1=1,
                 stride2=1,
                 stride3=1,
                 ):
        super(CnnEncoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, conv1_kernel, stride=stride1, padding=conv1_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, conv2_kernel, stride=stride2, padding=conv2_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, conv3_kernel, stride=stride3, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, conv3_kernel, stride=stride3, padding=conv3_kernel),
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
                 stride1=1,
                 stride2=1,
                 stride3=1,
                 ):
        super(CnnDecoder, self).__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(64, 32, conv3_kernel, stride=stride3, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, conv3_kernel, stride=stride3, padding=conv3_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 8, conv2_kernel, stride=stride2, padding=conv2_kernel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 1, conv1_kernel, stride=stride1, padding=conv1_kernel),
        )

    def forward(self, x):
        x = self.decoder_cnn(x)
        return x

#################################################################################################
# LSTM Stacked
#################################################################################################

class LstmEncoder(nn.Module):
    def __init__(self, seq_len=30, n_features=1, embedding_dim=64):
        super(LstmEncoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=5,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=5,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.n_features)
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.view(batch_size, self.embedding_dim)


class LstmDecoder(nn.Module):
    def __init__(self, seq_len=30, input_dim=1, n_features=1):
        super(LstmDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=5,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=5,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.repeat(self.seq_len, self.n_features) # todo testare se funziona con più feature
        x = x.view(batch_size, self.seq_len, self.input_dim)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.view(batch_size, self.seq_len, self.hidden_dim)

        return self.output_layer(x)
    