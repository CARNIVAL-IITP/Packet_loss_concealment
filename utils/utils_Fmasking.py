import operator
import os
import time
from os import makedirs

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import soundfile as sf
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy.random import default_rng
from pystoi import stoi
from scipy.signal import dlti
from scipy.signal._upfirdn import upfirdn
from scipy.signal.filter_design import cheby1
from scipy.signal.fir_filter_design import firwin
from scipy.signal.signaltools import filtfilt, lfilter, resample_poly
from tqdm import tqdm
from tqdm.auto import tqdm
from natsort import natsorted
from pesq import pesq as pesqq

from config import CONFIG
# from config_1 import CONFIG


rng = default_rng()


def decimate(x, q, ripple=0.05, n=None, ftype='iir', axis=-1, zero_phase=True):
    x = np.asarray(x)
    q = operator.index(q)

    if n is not None:
        n = operator.index(n)

    if ftype == 'fir':
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len
        b, a = firwin(n + 1, 1. / q, window='hamming'), 1.
    elif ftype == 'iir':
        if n is None:
            n = 8
        system = dlti(*cheby1(n, ripple, 0.8 / q))
        b, a = system.num, system.den
    elif isinstance(ftype, dlti):
        system = ftype._as_tf()  # Avoids copying if already in TF form
        b, a = system.num, system.den
    else:
        raise ValueError('invalid ftype')

    result_type = x.dtype
    if result_type.kind in 'bui':
        result_type = np.float64
    b = np.asarray(b, dtype=result_type)
    a = np.asarray(a, dtype=result_type)

    sl = [slice(None)] * x.ndim
    a = np.asarray(a)

    if a.size == 1:  # FIR case
        b = b / a
        if zero_phase:
            y = resample_poly(x, 1, q, axis=axis, window=b)
        else:
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = upfirdn(b, x, up=1, down=q, axis=axis)
            sl[axis] = slice(None, n_out, None)

    else:
        if zero_phase:
            y = filtfilt(b, a, x, axis=axis)
        else:
            y = lfilter(b, a, x, axis=axis)
        sl[axis] = slice(None, None, q)

    return y[tuple(sl)]


def frame(a, w, s, copy=True):
    if len(a) < w:
        return np.expand_dims(np.hstack((a, np.zeros(w - len(a)))), 0)

    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::s]

    if copy:
        return view.copy()
    else:
        return view


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def visualize(hr, lr, recon, path):
    sr = CONFIG.DATA.sr
    window_size = 1024
    window = np.hanning(window_size)

    stft_hr = librosa.core.spectrum.stft(hr, n_fft=window_size, hop_length=512, window=window)
    stft_hr = 2 * np.abs(stft_hr) / np.sum(window)

    stft_lr = librosa.core.spectrum.stft(lr, n_fft=window_size, hop_length=512, window=window)
    stft_lr = 2 * np.abs(stft_lr) / np.sum(window)

    stft_recon = librosa.core.spectrum.stft(recon, n_fft=window_size, hop_length=512, window=window)
    stft_recon = 2 * np.abs(stft_recon) / np.sum(window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 10))
    ax1.title.set_text('HR signal')
    ax2.title.set_text('LR signal')
    ax3.title.set_text('Reconstructed signal')

    canvas = FigureCanvas(fig)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_hr), ax=ax1, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_lr), ax=ax2, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_recon), ax=ax3, y_axis='linear', x_axis='time', sr=sr)
    mkdir_p(path)
    fig.savefig(os.path.join(path, 'spec.png'))


def SNR(x, ref):
    # Signal-to-noise ratio
    ref_pow = (ref**2).mean().mean() + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean().mean() + np.finfo('float32').eps
    snr_val = 10 * np.log10(ref_pow / dif_pow)
    return snr_val

def SNR2(y_true, y_pred):
    n_norm = np.mean((y_true - y_pred) ** 2)
    s_norm = np.mean(y_true ** 2)
    return 10 * np.log10((s_norm / n_norm) + 1e-8)

def SI_SDR(target, preds):
    EPS = 1e-8
    alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    si_sdr_value = (np.sum(target_scaled ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value

def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S

def LSD(x_hr, x_pr):
    S1_FULL = get_power(x_hr, nfft=2048)
    S2_FULL = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1_FULL - S2_FULL) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_HIGH = S1_FULL[-(len(S1_FULL) - 1) // 2:, :]
    S2_HIGH = S2_FULL[-(len(S2_FULL) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1_HIGH - S2_HIGH) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_LOW = S1_FULL[0 : ((len(S1_FULL) - 1) // 2) - 1, :]
    S2_LOW = S2_FULL[0 : ((len(S2_FULL) - 1) // 2) - 1, :]
    lsd_low = np.mean(np.sqrt(np.mean((S1_LOW - S2_LOW) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high, lsd_low


def compute_metrics(x_hr, pred_audio):
    fs = 16000
    snr = SNR(x_hr, pred_audio)
    lsd, lsd_high, lsd_low = LSD(x_hr, pred_audio)
    sisdr = SI_SDR(x_hr, pred_audio)
    py_stoi = stoi(x_hr, pred_audio, fs, extended=False)
    estoi = stoi(x_hr, pred_audio, fs, extended=True)
    pesq = pesqq(fs, x_hr, pred_audio, 'wb')
    return np.array([py_stoi, estoi, snr, lsd, lsd_high, lsd_low, pesq, sisdr])


def overlap_add(x, win_len, hop_size, target_shape):
    # target.shape = (B, C, seq_len)
    # x.shape = (B*n_chunks, C, win_len) , n_chunks = (seq_len - hop_size)/(win_len - hop_size)
    bs, channels, seq_len = target_shape
    hann_windows = torch.ones(x.shape, device=x.device) * torch.hann_window(win_len, device=x.device)
    hann_windows[0, :, :hop_size] = 1
    hann_windows[-1, :, -hop_size:] = 1
    x *= hann_windows
    x = x.permute(1, 0, 2).reshape(bs * channels, -1, win_len).permute(0, 2, 1)  # B*C, win_len, n_chunks
    fold = torch.nn.Fold(output_size=(1, seq_len), kernel_size=(1, win_len), stride=(1, hop_size))
    x = fold(x)  # B*C, 1, 1, seq_len
    x = x.reshape(channels, bs, seq_len).permute(1, 0, 2)  # B, C, seq_len
    return x

def load_txt(target_root, txt_list):
    target = []
    with open(txt_list) as f:
        for line in f:
            target.append(os.path.join(target_root, line.strip('\n')))
    target = list(set(target))
    target = natsorted(target)
    return target

def evaluate_dataset(model, test_loader, sample_path,  version, cpu = False, single = False, save = False):
    window_size, stride, sr = test_loader.dataset.window, test_loader.dataset.stride, test_loader.dataset.sr

    results = []  # lsd, lsd_high, sisdr
    latency = []

    data_dir = CONFIG.DATA.data_dir
    name = CONFIG.DATA.dataset
    target_root = data_dir[name]['root']
    clean_txt_list = data_dir[name]['test_clean']
    clean_data_list = load_txt(target_root,clean_txt_list)
    out_path = CONFIG.TEST.out_dir  + '/version_' + str(version) + '/' + CONFIG.TASK.task 
    makedirs(out_path, exist_ok = True)

    for i, (x_lr, x_hr, inp) in enumerate(tqdm(test_loader)):
        x_hr = x_hr.numpy()[0, :]
        inp = inp.numpy()[0, :]
        if cpu:
            device = torch.device('cpu')
            x_lr = torch.Tensor(x_lr).to(device)
            start = time.time()
            pred = model(x_lr[0])
            latency.append(time.time() - start)
        else:
            start = time.time()
            pred = model(x_lr.cuda(device=0)[0])
            latency.append(time.time() - start)

        pred = overlap_add(pred, window_size, stride, (1, 1, len(x_hr)))  # batch_size=1, 1 channel
        pred = torch.squeeze(pred).detach().cpu().numpy()
        ret = compute_metrics(x_hr, pred)
        results.append(ret)

        head, tail = os.path.split(clean_data_list[i])
        # tail = tail.split('.wav')[0]

        if save:
            sf.write(os.path.join(out_path, tail), pred, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, tail + '_pr.wav'), pred, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, tail + '_lr.wav'), inp, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, tail + '_hr.wav'), x_hr, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, str(i) + '_pr.wav'), pred, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, str(i) + 'lr.wav'), inp, samplerate=sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path, str(i) + '_hr.wav'), x_hr, samplerate=sr, subtype='PCM_16')
       
    print("Pytorch Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))
    print("max inference time = {} ms".format(format(max(latency) * 1000, '.2f')))
    print("number of iterations: ", len(latency))

    # txt_name = './result/SITEC/infrence_time_version' + str(version) + '_torch_gpu.txt'
    # line = "PyTorch {} Inference time = {} ms".format('gpu', format(sum(latency) * 1000 / len(latency), '.2f'))
    # file = open(txt_name, "w")
    # file.write(str(line))
    # file.write("\n")
    # file.close()
    # print(txt_name)

    results = np.array(results)
    return np.vstack((results.mean(0), results.std(0))).T

# def evaluate_dataset(model, test_loader, sample_path, eval_input=False):
#     window_size, stride, sr = test_loader.dataset.window, test_loader.dataset.stride, test_loader.dataset.sr
#     results = []  # lsd, lsd_high, sisdr
#     for i, (x_lr, x_hr, inp) in enumerate(tqdm(test_loader)):
#         x_hr = x_hr.numpy()[0, :]
#         inp = inp.numpy()[0, :]
#         if not eval_input:
#             pred = model(x_lr.cuda(device=0)[0])
#             pred = overlap_add(pred, window_size, stride, (1, 1, len(x_hr)))  # batch_size=1, 1 channel
#             pred = torch.squeeze(pred).detach().cpu().numpy()
#             if i in [1, 5, 7]:
#                 path = os.path.join(sample_path, 'sample_' + str(i))
#                 visualize(x_hr, inp, pred, path)
#                 sf.write(os.path.join(path, 'recon.wav'), pred, samplerate=sr, subtype='PCM_16')
#                 sf.write(os.path.join(path, 'low_rate.wav'), inp, samplerate=sr, subtype='PCM_16')
#                 sf.write(os.path.join(path, 'high_rate.wav'), x_hr, samplerate=sr, subtype='PCM_16')
#             ret = compute_metrics(x_hr, pred)
#         else:
#             ret = compute_metrics(x_hr, inp)
#         results.append(ret)
#     results = np.array(results)
#     return np.vstack((results.mean(0), results.std(0))).T