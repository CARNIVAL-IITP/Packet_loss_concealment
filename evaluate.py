import operator, os, librosa, glob
from  librosa import display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf 
import tqdm
from pystoi import stoi
from tqdm.auto import tqdm
from natsort import natsorted
from os import makedirs
from pesq import pesq as pesqq

from algorithmLib import compute_audio_quality


input_clean_path = '/home/donghyun2/Research/TUNet/TUNet-plc/output/plc-challenge/hr/'
input_enhanced_path = '/home/donghyun2/Research/TUNet/TUNet-plc/oRutput/plc-challenge/lr/'

txt_name = '/home/donghyun2/Research/TUNet/TUNet-plc/result/plc-challenge/plc-challenge_lr_result_pesq_lsd-low.txt'
# makedirs(r_path, exist_ok =  True)

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
    S = librosa.stft(y = x, n_fft = nfft, hop_length=512)
    S = np.log(np.abs(S) ** 2 + 1e-8) 
    return S

# def LSD(x_hr, x_pr):
#     S1 = get_power(x_hr, nfft=2048)
#     S2 = get_power(x_pr, nfft=2048)
#     lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
#     S1 = S1[-(len(S1) - 1) // 2:, :]
#     S2 = S2[-(len(S2) - 1) // 2:, :]
#     lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
#     return lsd, lsd_high

def LSD(x_hr, x_pr):
    # a = [101, 102, 'b', 'l', 'o', 103, 104,105,106]
    # b = [101, 102, 'b', 'l', 'o', 103, 104,105,106]
    # a = [a,b]
    # print(len(a))
    # a_high = a[-(len(a - 1)) // 2:, :]
    # a_low = a[0:(len(a - 1)) // 2, :]
    # print(a)
    # print(a_high)
    # print(a_low)
    # exit()
    S1_FULL = get_power(x_hr, nfft=2048)
    S2_FULL = get_power(x_pr, nfft=2048)
    # print(len(S1_FULL))
    # print(len(S2_FULL))
    # print(S1_FULL)
    # exit()
    lsd = np.mean(np.sqrt(np.mean((S1_FULL - S2_FULL) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_HIGH = S1_FULL[-(len(S1_FULL) - 1) // 2:, :]
    S2_HIGH = S2_FULL[-(len(S2_FULL) - 1) // 2:, :]
    # print(S1_HIGH)
    # print(len(S1_HIGH))
    # print(len(S2_HIGH))
    # print(S2_HIGH)
    # exit()
    # print(S1_HIGH)
    lsd_high = np.mean(np.sqrt(np.mean((S1_HIGH - S2_HIGH) ** 2 + 1e-8, axis=-1)), axis=0)
    S1_LOW = S1_FULL[0 : (len(S1_FULL) - 1) // 2, :]
    S2_LOW = S2_FULL[0 : (len(S2_FULL) - 1) // 2, :]
    # print(len(S1_LOW))
    # print(len(S2_LOW))
    # print(S1_LOW)
    # exit()
    lsd_low = np.mean(np.sqrt(np.mean((S1_LOW - S2_LOW) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high, lsd_low

def compute_metrics(x_hr, pred_audio, fs):
    snr = SNR(x_hr, pred_audio)
    lsd, lsd_high, lsd_low = LSD(x_hr, pred_audio)
    sisdr = SI_SDR(x_hr, pred_audio)
    py_stoi = stoi(x_hr, pred_audio, fs, extended=False)
    estoi = stoi(x_hr, pred_audio, fs, extended=True)
    pesq = pesqq(fs, x_hr, pred_audio, 'wb')
    return np.array([py_stoi, estoi, snr, lsd, lsd_high, lsd_low, pesq, sisdr])

def evaluate_dataset(input_clean_path, input_enhanced_path):
    results = []

    hr_files = os.listdir(input_clean_path)
    # hr_files.sort()
    hr_files = natsorted(hr_files)
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(input_clean_path + hr_file)

    lr_files = os.listdir(input_enhanced_path)
    # lr_files.sort()
    lr_files = natsorted(lr_files)
    lr_file_list = []
    for lr_file in lr_files:
        lr_file_list.append(input_enhanced_path + lr_file)
  
    # file_num = len(hr_file_list)
    # assert file_num == len(lr_file_list)

    all_files = glob.glob(input_clean_path +"*.wav")
    all_files = natsorted(all_files)
    file_length = len(all_files)

    # for filecounter in tqdm(range(0,file_length)):
    for i in tqdm(range (800)):

        # print(lr_file_list[i])
        # print(hr_file_list[i])

        # wav_clean_name= all_files[filecounter]
        # wav_enhanced_name = all_files[filecounter].replace(input_clean_path, input_enhanced_path)

        # wav_clean_name = 'clean_' + str(i+1) + '.wav'
        # wav_enhanced_name = 'output_' + str(i+1) + '.wav'

        # wav_clean_name = str(i) + '.2.hr.wav'
        # wav_enhanced_name = str(i) + '.2.pr.wav'
        # x_hr, fs = sf.read(wav_clean_name)
        # pred , fs = sf.read(wav_enhanced_name)

        # x_hr, fs = sf.read(input_clean_path + wav_clean_name)
        # pred, fs = sf.read(input_enhanced_path + wav_enhanced_name)

        x_hr, fs = sf.read(hr_file_list[i])
        pred, fs = sf.read(lr_file_list[i])

        # wav_clean_name = str(i) + '.2.hr.wav'
        # wav_enhanced_name = str(i) + '.2.lr.wav'
        # wav_clean_name = 'clean_test_' + str(i+1) + '.wav'
        # wav_enhanced_name = 'output_' + str(i+1) + '.wav'

        # print(wav_clean_name)
        # print(wav_enhanced_name)
        
        # x_hr, fs = sf.read(input_clean_path + wav_clean_name)
        # pred, fs = sf.read(input_enhanced_path + wav_enhanced_name)

        ret = compute_metrics(x_hr, pred, fs)
        results.append(ret)
    
    results = np.array(results)
    
    return np.vstack((results.mean(0), results.std(0))).T

res = evaluate_dataset(input_clean_path, input_enhanced_path)
print("Evaluate-- STOI: {} ESTOI: {} SNR: {} LSD: {} LSD-HF: {} LSD-LF: {} PESQ: {} SI-SDR: {}".format(res[0], 
        res[1], res[2], res[3], res[4], res[5], res[6], res[7]))
# np.savetxt("./result_stoi/eval_r2_atafilm_t.txt", res)
file = open(txt_name, "w")
f = "{0:<16} {1:<16}"
file.write(f.format("Mean", "Std")+"\n")
file.write("---------------------------------\n")
file.write(str(res))
file.write("\n")
file.write("---------------------------------\n")
metric = "STOI, ESTOI, SNR, LSD, LSD-HF, LSD-LF, PESQ, SI-SDR"
file.write(metric)
file.close()
print(txt_name)

# file = open("./result_stoi/eval_r2_atafilm_tt.txt", "w")
# f = "{0:<16} {1:<16}"
# file.write(f.format("Noise", "PESQ")+"\n")
# file.write("---------------------------------\n")
# file.write(f.format("Avg.", "%.2f +- %.2f\n" % (res[0].astype(int), res[1].astype(int))))
# file.close()

