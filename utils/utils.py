import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import CONFIG


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def visualize(target, input, recon, path):
    sr = CONFIG.DATA.sr
    window_size = 1024
    window = np.hanning(window_size)

    stft_hr = librosa.core.spectrum.stft(target, n_fft=window_size, hop_length=512, window=window)
    stft_hr = 2 * np.abs(stft_hr) / np.sum(window)

    stft_lr = librosa.core.spectrum.stft(input, n_fft=window_size, hop_length=512, window=window)
    stft_lr = 2 * np.abs(stft_lr) / np.sum(window)

    stft_recon = librosa.core.spectrum.stft(recon, n_fft=window_size, hop_length=512, window=window)
    stft_recon = 2 * np.abs(stft_recon) / np.sum(window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 10))
    ax1.title.set_text('Target signal')
    ax2.title.set_text('Lossy signal')
    ax3.title.set_text('Reconstructed signal')

    canvas = FigureCanvas(fig)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_hr), ax=ax1, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_lr), ax=ax2, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_recon), ax=ax3, y_axis='linear', x_axis='time', sr=sr)
    mkdir_p(path)
    fig.savefig(os.path.join(path, 'spec.png'))


def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


# origin
# def LSD(x_hr, x_pr):
#     S1 = get_power(x_hr, nfft=2048)
#     S2 = get_power(x_pr, nfft=2048)
#     lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
#     S1 = S1[-(len(S1) - 1) // 2:, :]
#     S2 = S2[-(len(S2) - 1) // 2:, :]
#     lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
#     return lsd, lsd_high

def LSD(x_hr, x_pr):
    S1_FULL = get_power(x_hr, nfft=2048)
    S2_FULL = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1_FULL - S2_FULL) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_HIGH = S1_FULL[-(len(S1_FULL) - 1) // 2:, :]
    S2_HIGH = S2_FULL[-(len(S2_FULL) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1_HIGH - S2_HIGH) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_LOW = S1_FULL[0 : (len(S1_FULL) - 1) // 2, :]
    S2_LOW = S2_FULL[0 : (len(S2_FULL) - 1) // 2, :]
    lsd_low = np.mean(np.sqrt(np.mean((S1_LOW - S2_LOW) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high, lsd_low
