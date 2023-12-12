import librosa
import pytorch_lightning as pl
import torch
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, apply_reduction, SpectralConvergenceLoss, STFTMagnitudeLoss

from config import CONFIG


class STFTLossDDP(STFTLoss):
    def __init__(self,
                 fft_size=1024,
                 hop_size=256,
                 win_length=1024,
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 eps=1e-8,
                 output="loss",
                 reduction="mean",
                 device=None):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.device = device

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction)

        # setup mel filterbank
        if self.scale == "mel":
            assert (sample_rate is not None)  # Must set sample rate to use mel scale
            assert (n_bins <= fft_size)  # Must be more FFT bins than Mel bins
            fb = librosa.filters.mel(sample_rate, fft_size, n_mels=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)
        elif self.scale == "chroma":
            assert (sample_rate is not None)  # Must set sample rate to use chroma scale
            assert (n_bins <= fft_size)  # Must be more FFT bins than chroma bins
            fb = librosa.filters.chroma(sample_rate, fft_size, n_chroma=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    def compressed_loss(self, x, y, alpha=None):
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        if alpha is not None:
            x_mag = x_mag ** alpha
            y_mag = y_mag ** alpha

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb.to(x_mag.device), x_mag)
            y_mag = torch.matmul(self.fb.to(y_mag.device), y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0

        # combine loss terms
        loss = (self.w_sc * sc_loss) + (self.w_log_mag * mag_loss) + (self.w_lin_mag * lin_loss)
        loss = apply_reduction(loss, reduction=self.reduction)
        return loss

    def forward(self, x, y):
        return self.compressed_loss(x, y, 0.3)


class MRSTFTLossDDP(MultiResolutionSTFTLoss):
    def __init__(self,
                 fft_sizes=(1024, 2048, 512),
                 hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240),
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 **kwargs):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossDDP(fs,
                                             ss,
                                             wl,
                                             window,
                                             w_sc,
                                             w_log_mag,
                                             w_lin_mag,
                                             w_phs,
                                             sample_rate,
                                             scale,
                                             n_bins,
                                             scale_invariance,
                                             **kwargs)]


class Loss(pl.LightningModule):
    def __init__(self):
        super(Loss, self).__init__()
        self.stft_loss = MRSTFTLossDDP(sample_rate=CONFIG.DATA.sr, device="cpu", w_log_mag=0.0, w_lin_mag=1.0)
        self.window = torch.sqrt(torch.hann_window(CONFIG.DATA.window_size))

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        wave_x = torch.istft(torch.view_as_complex(x.contiguous()), CONFIG.DATA.window_size, CONFIG.DATA.stride,
                             window=self.window.to(x.device))
        wave_y = torch.istft(torch.view_as_complex(y.contiguous()), CONFIG.DATA.window_size, CONFIG.DATA.stride,
                             window=self.window.to(y.device))
        loss = self.stft_loss(wave_x, wave_y)
        return loss
