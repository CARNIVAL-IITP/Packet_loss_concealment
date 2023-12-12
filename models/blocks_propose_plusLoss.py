import librosa
import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np


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
        # print('mlp 4', x.shape, inter.shape) # state 있네. mlp 4 torch.Size([32, 1, 384]) torch.Size([32, 1, 384])
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
        # print('en 3', x.shape) # en 3 torch.Size([32, 2, 160, 1]) B,C,F,T 인듯 T가 1씩 들어가다니..
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


class AcousticEstimator(torch.nn.Module):
    def __init__(self):
        super(AcousticEstimator, self).__init__()
        self.lstm = torch.nn.LSTM(514, 256, 3, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 25)
        self.act = torch.nn.ReLU()
        # self.weight = nn.Parameter(torch.rand(1,1,25), requires_grad=False) # False로 하는게 맞나... # random weight > 57 > 198.pt
        # 위에 random weight grad= True로 주면 어케됨? 궁금... 해보자 > 59
        self.weight = nn.Parameter(torch.rand(1, 1, 25), requires_grad=False)
        self.soft = nn.Softmax(dim=2)

    def forward(self, A0):
        # print('weight', self.weight)
        A1, _ = self.lstm(A0)
        Z1 = self.linear1(A1)
        A2 = self.act(Z1)
        Z2 = self.linear2(A2)
        # Z3 = Z2 * self.weight # random weight > 57
        ratio = Z2.mean()*self.soft(self.weight)
        Z3 = Z2 * ratio
        # print('ratio',self.soft(self.weight), ratio)
        return Z3 #Z2

class AcousticLoss(torch.nn.Module):
    def __init__(self, loss_type: str, acoustic_model_path: str, paap: bool = False, paap_weight_path: str = None,
                 device='cuda'):
        """
        Args:
            loss_type (str):
                Must be one of the following 4 options: ["l2", "l1", "frame_energy_weighted_l2", "frame_energy_weighted_l1"]
            acoustic_model_path (str):
                Path to the pretrained temporal acoustic parameter estimator model checkpoint.
            paap (bool):
                True for use PAAPLoss, False for use TAPLoss.
            paap_weight_path (str):
                Path to the Paap weight .npy file
        """
        super(AcousticLoss, self).__init__()
        self.device = device
        self.paap = paap
        self.estimate_acoustics = AcousticEstimator()
        self.loss_type = loss_type
        if self.loss_type == "l2":
            self.l2 = torch.nn.MSELoss()
        elif self.loss_type == "l1":
            self.l1 = torch.nn.L1Loss()
        if paap:
            if paap_weight_path is None:
                raise ValueError("PAAP weight path is not provided")
            else:
                self.paap_weight = torch.from_numpy(np.load(paap_weight_path)).to(device)

        #print(torch.load(acoustic_model_path, map_location=device))
        #exit()
        model_state_dict = torch.load(acoustic_model_path, map_location=device)['model_state_dict'] # original need 'model_state_dict'
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
        for param in self.estimate_acoustics.parameters():
            param.requires_grad = False

    def __call__(self, clean_waveform, enhan_waveform, mode="train"):
        return self.forward(clean_waveform, enhan_waveform, mode)

    def forward(self, clean_waveform: torch.FloatTensor, enhan_waveform: torch.FloatTensor,
                mode: str) -> torch.FloatTensor:

        """
        Args:
            clean_waveform (torch.FloatTensor)：
                Tensor of clean waveform with shape (B, T * sr).
            enhan_waveform (torch.FloatTensor)：
                Tensor of enhanced waveform with shape (B, T * sr).
            mode (str) :
                'train' or 'eval'
        Returns:
            acoustic_loss (torch.FloatTensor):
                Loss value corresponding to the selected loss type.
        """

        if mode == "train":
            self.estimate_acoustics.train()
        elif mode == "eval":
            self.estimate_acoustics.eval()
        else:
            raise ValueError("Invalid mode, must be either 'train' or 'eval'.")

        clean_spectrogram = self.get_stft(clean_waveform)
        enhan_spectrogram, enhan_st_energy = self.get_stft(enhan_waveform, return_short_time_energy=True)
        clean_acoustics = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics = self.estimate_acoustics(enhan_spectrogram)
        if self.paap:
            """
            paap_weight ==> (26, 40) 
            acoustics ==> (B, T * sr, 25), expand last dimension by 1 for bias
            """
            clean_acoustics = torch.cat((clean_acoustics, torch.ones(clean_acoustics.size(dim=0), \
                                                                     clean_acoustics.size(dim=1), 1,
                                                                     device=self.device)),
                                        dim=-1)  # acoustics ==> (B, T, 26)

            enhan_acoustics = torch.cat((enhan_acoustics, torch.ones(enhan_acoustics.size(dim=0), \
                                                                     enhan_acoustics.size(dim=1), 1,
                                                                     device=self.device)),
                                        dim=-1)  # acoustics ==> (B, T, 26)

            clean_acoustics = clean_acoustics @ self.paap_weight  # acoustics ==> (B, T, 40)
            enhan_acoustics = enhan_acoustics @ self.paap_weight  # acoustics ==> (B, T, 40)
        """
        loss_type must be one of the following 4 options:
        ["l2", "l1", "frame_energy_weighted_l2", "frame_energy_weighted_l1"]
        """
        if self.loss_type == "l2":
            acoustic_loss = self.l2(enhan_acoustics, clean_acoustics)
        elif self.loss_type == "l1":
            acoustic_loss = self.l1(enhan_acoustics, clean_acoustics)
        elif self.loss_type == "frame_energy_weighted_l2":
            acoustic_loss = torch.mean(((torch.sigmoid(enhan_st_energy) ** 0.5).unsqueeze(dim=-1) \
                                        * (enhan_acoustics - clean_acoustics)) ** 2)
        elif self.loss_type == "frame_energy_weighted_l1":
            acoustic_loss = torch.mean(torch.sigmoid(enhan_st_energy).unsqueeze(dim=-1) \
                                       * torch.abs(enhan_acoustics - clean_acoustics))
        else:
            raise ValueError("Invalid loss_type {}".format(self.loss_type))

        return acoustic_loss

    def get_stft(self, wav: torch.FloatTensor, return_short_time_energy: bool = False) -> torch.FloatTensor:
        """
        Args:
            wav (torch.FloatTensor):
                Tensor of waveform of shape: (B, T * sr).
            return_short_time_energy (bool):
                True to return both complex spectrogram and short-time energy.
        Returns:
            spec (torch.FloatTensor):
                Real value representation of complex spectrograms, real part \
                and imag part alternate along the frequency axis.
            st_energy (torch.FloatTensor):
                Short-time energy calculated in the frequency domain using Parseval's theorem.
        """
        self.nfft = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length,
                          return_complex=False)  # Rectangular window with window length = 32ms, hop length = 10ms
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1,
                                                2 * (self.nfft // 2 + 1))  # spec ==> (B, T * sr, 2 * (nfft / 2 + 1))

        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real ** 2 + spec_imag ** 2, dim=1), 2 / self.nfft)
            return spec.float(), st_energy.float()
        else:
            return spec.float()