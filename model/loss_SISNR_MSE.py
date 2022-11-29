import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import math
from asteroid_filterbanks import transforms
from asteroid.losses import SingleSrcPMSQE, PITLossWrapper
from asteroid_filterbanks import STFTFB, Encoder

def mse_loss():
    def loss_function(est, label, nframes):

        # [B, 2, T, F]

        EPSILON = 1e-7
        with torch.no_grad():
            mask_for_loss_list = []
            # make mask
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[3], dtype=torch.float32))
            # output: [B T F]
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


def MSE_LOSS():
    def loss_function(est, label):
        mse_loss = F.mse_loss(est, label, reduction='mean')
        return mse_loss
    return loss_function

pmsqe_stft = Encoder(STFTFB(kernel_size=32, n_filters=32, stride=16))
pmsqe_loss = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_mtx')
def MSE_PMSQE():
    def loss_function(inputs, labels):
        # ref_wav = labels.reshape(-1, 3, 16000)
        # est_wav = inputs.reshape(-1, 3, 16000)
        ref_wav = labels.reshape(labels.shape[0], 1, -1)
        est_wav = inputs.reshape(inputs.shape[0], 1, -1)
        ref_wav = ref_wav.cpu()
        est_wav = est_wav.cpu()

        # ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
        # est_spec = transforms.take_mag(pmsqe_stft(est_wav))
        ref_spec = transforms.mag(pmsqe_stft(ref_wav))
        est_spec = transforms.mag(pmsqe_stft(est_wav))
        print('이건 나와?', ref_spec.shape, est_spec.shape) #이건 나와? torch.Size([11408, 17, 19]) torch.Size([11408, 17, 19])
        exit()
        loss = pmsqe_loss(ref_spec, est_spec)

        loss = loss.cuda()
        return loss
    return loss_function

def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

def sdr(s1, s2, eps=1e-8):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = 10 * torch.log10(sn**2 / (sn_m_shn**2 + eps))
    return torch.mean(sdr_loss)

def MSE_SISNR():
    def loss_function(inputs, labels):
        snr_loss = -(si_snr(inputs, labels))
        mse_loss = F.mse_loss(inputs, labels, reduction='mean')

        r1 = 1
        r2 = 10000 #100 (base)
        r = r1 + r2

        loss = (r1 * snr_loss + r2 * mse_loss) / r

        return loss
    return loss_function

def MSE_SDR():
    def loss_function(inputs, labels):
        sdr_loss = -(sdr(inputs, labels))
        mse_loss = F.mse_loss(inputs, labels, reduction='mean')

        r1 = 1
        r2 = 100
        r = r1 + r2

        loss = (r1 * sdr_loss + r2 * mse_loss) / r
        return loss
    return loss_function