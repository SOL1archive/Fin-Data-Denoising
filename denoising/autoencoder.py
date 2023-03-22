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
# LSTM: Seq2Seq
#################################################################################################

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat(self.seq_len, self.n_features) # todo testare se funziona con più feature
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


######
# MAIN
######


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, device='cuda', batch_size=32):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

'''
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.value_vector = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.repeat(encoder_outputs.shape[1], 1, 1)
        
        attn_input = torch.cat((hidden, encoder_outputs), dim=2)
        
        attn_weights = torch.tanh(self.attention_layer(attn_input))
        attn_weights = self.value_vector(attn_weights).squeeze()

        attn_probs = torch.softmax(attn_weights, dim=1)
        context_vector = torch.bmm(attn_probs.unsqueeze(1), encoder_outputs)
        
        return context_vector, attn_probs

class LstmAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LstmAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=1, batch_first=True)
        
        self.attention = Attention(hidden_size)
        
    def forward(self, x):
        encoder_output, (hidden_state, cell_state) = self.encoder(x)
        
        decoder_hidden = hidden_state
        decoder_cell = cell_state
        
        

        return decoder_output
'''