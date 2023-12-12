import librosa
import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from torch import nn


class Aff(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class FeedForward(pl.LightningModule):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLPBlock(pl.LightningModule):

    def __init__(self, dim, mlp_dim, dropout=0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.inter = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1,
                             bidirectional=False, batch_first=True)
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, state=None):
        # print('mlp 1', x.shape) # mlp 1 torch.Size([32, 1, 384])
        x = self.pre_affine(x)
        # print('mlp 2', x.shape) #mlp 2 torch.Size([32, 1, 384])
        if state is None:
            inter, _ = self.inter(x)
        else:
            inter, state = self.inter(x, (state[0], state[1]))
        # print('mlp 4', x.shape, inter.shape) 
        x = x + self.gamma_1 * inter
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        if state is None:
            return x
        state = torch.stack(state, 0)
        return x, state


class Encoder(pl.LightningModule):

    def __init__(self, in_dim=320, dim=384, depth=4, mlp_dim=768):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c f t -> b t (c f)'),
            nn.Linear(in_dim, dim),
            nn.GELU()
        )
        # self.encoder.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c f t -> b t (c f)'),
        #     nn.Linear(in_dim, dim),
        #     nn.GELU()
        # )

        self.mlp_blocks = nn.ModuleList([])
        # self.encoder.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPBlock(self.dim, mlp_dim, dropout=0.15))
            # self.encoder.mlp_blocks.append(MLPBlock(self.dim, mlp_dim, dropout=0.15))

        self.affine = nn.Sequential(
            Aff(self.dim),
            nn.Linear(dim, in_dim),
            Rearrange('b t (c f) -> b c f t', c=2),
        )
        # self.encoder.affine = nn.Sequential(
        #     Aff(self.dim),
        #     nn.Linear(dim, in_dim),
        #     Rearrange('b t (c f) -> b c f t', c=2),
        # )

    def forward(self, x_in, states=None):
        # print('encoder', x_in.shape) # encoder torch.Size([32, 2, 160, 1])
        x = self.to_patch_embedding(x_in)
        if states is not None:
            out_states = []
        for i, mlp_block in enumerate(self.mlp_blocks):
            if states is None:
                x = mlp_block(x)
            else:
                x, state = mlp_block(x, states[i])
                out_states.append(state)
        # print('en 2', x.shape) # en 2 torch.Size([32, 1, 384])
        x = self.affine(x)
        # print('en 3', x.shape) # en 3 torch.Size([32, 2, 160, 1]) 
        x = x + x_in
        if states is None:
            return x
        else:
            return x, torch.stack(out_states, 0)


class Predictor(pl.LightningModule):  # mel #pl.LightningModule
    def __init__(self, window_size=320, sr=16000, lstm_dim=256, lstm_layers=3, n_mels=64):
        super(Predictor, self).__init__()
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.lstm_dim = lstm_dim
        self.n_mels = n_mels
        self.lstm_layers = lstm_layers

        fb = librosa.filters.mel(sr=sr, n_fft=self.window_size, n_mels=self.n_mels)[:, 1:]
        self.fb = torch.from_numpy(fb).unsqueeze(0).unsqueeze(0)
        self.lstm = nn.LSTM(input_size=self.n_mels, hidden_size=self.lstm_dim, bidirectional=False,
                            num_layers=self.lstm_layers, batch_first=True)
        self.expand_dim = nn.Linear(self.lstm_dim, self.n_mels)
        self.inv_mel = nn.Linear(self.n_mels, self.hop_size)

    def forward(self, x, state=None):  # B, 2, F, T
        self.fb = self.fb.to(x.device)
        x = torch.log(torch.matmul(self.fb, x) + 1e-8)
        B, C, F, T = x.shape
        x = x.reshape(B, F * C, T)
        x = x.permute(0, 2, 1)
        if state is None:
            x, _ = self.lstm(x)
        else:
            x, state = self.lstm(x, (state[0], state[1]))
        x = self.expand_dim(x)
        x = torch.abs(self.inv_mel(torch.exp(x)))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, -1, T)
        if state is None:
            return x
        else:
            return x, torch.stack(state, 0)

class RI_Predictor(nn.Module):  # mel
    def __init__(self, window_size=320, sr=16000, lstm_dim=256, lstm_layers=3):
        super(RI_Predictor, self).__init__()
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.lstm_dim = lstm_dim
        # self.n_mels = n_mels
        self.lstm_layers = lstm_layers

        # fb = librosa.filters.mel(sr=sr, n_fft=self.window_size, n_mels=self.n_mels)[:, 1:]
        # self.fb = torch.from_numpy(fb).unsqueeze(0).unsqueeze(0)
        self.lstm = nn.LSTM(input_size=self.window_size, hidden_size=self.lstm_dim, bidirectional=False,
                            num_layers=self.lstm_layers, batch_first=True)
        self.expand_dim = nn.Linear(self.lstm_dim, self.window_size)
        # self.inv_mel = nn.Linear(self.n_mels, self.hop_size)

    def forward(self, x, state=None):  # B, 2, F, T
        # self.fb = self.fb.to(x.device)
        # x = torch.log(torch.matmul(self.fb, x) + 1e-8)
        B, C, F, T = x.shape
        x = x.reshape(B, F * C, T)
        x = x.permute(0, 2, 1)
        if state is None:
            x, _ = self.lstm(x)
        else:
            x, state = self.lstm(x, (state[0], state[1]))
        x = self.expand_dim(x)
        # x = torch.abs(self.inv_mel(torch.exp(x)))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, -1, T)
        if state is None:
            return x
        else:
            return x, torch.stack(state, 0)
