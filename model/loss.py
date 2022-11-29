import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import math

def mse_loss():
    def loss_function(est, label, nframes):
        # [B, 2, T, F]

        EPSILON = 1e-7
        with torch.no_grad():
            mask_for_loss_list = []
            # make mask
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[3], dtype=torch.float32))

            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda()
            mask_for_loss = mask_for_loss.unsqueeze_(1)
            mask_for_loss_complex = torch.cat((mask_for_loss, mask_for_loss), 1)

        masked_est = est * mask_for_loss_complex
        masked_label = label * mask_for_loss_complex
        loss = ((masked_est - masked_label) ** 2).sum() / mask_for_loss_complex.sum() + EPSILON
        return loss
    return loss_function
def mse_loss_howl():
    def loss_function(est,label, nframes):

        loss = ((label - est) ** 2).sum() / (est.shape[0] * est.shape[1] * est.shape[2])
        return loss
    return loss_function

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)
def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

# generate Mel filter bank
def melFilterBank(numCoeffs, fftSize=None):
    minHz = 0
    maxHz = 512 / 2  # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = 320
    else:
        numFFTBins = int(fftSize / 2) + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    melRange = np.array(range(numCoeffs + 2))
    melRange = melRange.astype(np.float32)

    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = math.floor(numFFTBins * melCenterFilters[i] / maxHz)

    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, numFFTBins))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)

        startRange = int(melCenterFilters[i - 1])
        midRange = int(melCenterFilters[i])
        endRange = int(melCenterFilters[i + 1])

        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))

        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat

FFT_SIZE = 512

# multi-scale MFCC distance
MEL_SCALES = [16, 32, 64]  # for LMS
DEVICE = torch.device("cuda")

def perceptual_transform(x):

    MEL_FILTERBANKS = []
    for scale in MEL_SCALES:
        filterbank_npy = melFilterBank(scale, FFT_SIZE).transpose()
        torch_filterbank_npy = torch.from_numpy(filterbank_npy).type(torch.FloatTensor)
        MEL_FILTERBANKS.append(torch_filterbank_npy.to(DEVICE))

    transforms = []

    powerSpectrum = x.view(-1, FFT_SIZE // 2 + 1)
    powerSpectrum = 1.0 / FFT_SIZE * powerSpectrum

    for filterbank in MEL_FILTERBANKS:
        filteredSpectrum = torch.mm(powerSpectrum, filterbank)
        filteredSpectrum = torch.log(filteredSpectrum + 1e-7)
        transforms.append(filteredSpectrum)

    return transforms

class rmse(torch.nn.Module):
    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = torch.sqrt(mse + 1e-7)

        return torch.mean(rmse)

# perceptual loss function
class perceptual_distance(torch.nn.Module):

    def __init__(self):
        super(perceptual_distance, self).__init__()

    def forward(self, y_true, y_pred):
        rmse_loss = rmse()

        pvec_true = perceptual_transform(y_true)
        pvec_pred = perceptual_transform(y_pred)

        distances = []
        for i in range(0, len(pvec_true)):
            error = rmse_loss(pvec_pred[i], pvec_true[i])
            error = error.unsqueeze(dim=-1)
            distances.append(error)
        distances = torch.cat(distances, axis=-1)

        loss = torch.mean(distances, axis=-1)
        return torch.mean(loss)

get_mel_loss = perceptual_distance()
def get_array_mel_loss(clean_array, est_array):
    array_mel_loss = 0
    for i in range(len(clean_array)):
        mel_loss = get_mel_loss(clean_array[i], est_array[i])
        array_mel_loss += mel_loss

    avg_mel_loss = array_mel_loss / len(clean_array)
    return avg_mel_loss

def MSE_LMS():
    def loss_function(est, label):
        mse_loss = F.mse_loss(est, label, reduction='mean')

        #for mel loss calculation
        label_real = label[:,0,:]
        label_imag = label[:,1,:]
        label_mag = torch.sqrt(label_real**2 + label_imag**2 + 1e-7)

        est_real = est[:,0,:]
        est_imag = est[:,1,:]
        est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-7)
        mel_loss = get_array_mel_loss(label_mag, est_mag)

        r1 = 1e+3
        r2 = 1
        r = r1 + r2
        loss = (r1 * mse_loss + r2 * mel_loss) / r
        return loss
    return loss_function

def MSE_LOSS():
    def loss_function(est, label):
        mse_loss = F.mse_loss(est, label, reduction='mean')
        return mse_loss
    return loss_function
