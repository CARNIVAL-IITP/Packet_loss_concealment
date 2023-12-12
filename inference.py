import os
import random

import librosa
import numpy as np
import torch
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG
from utils.utils import decimate, frame

from natsort import natsorted
from os import makedirs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# model_path = "/home/donghyun/Research/AFILM/vctk_single_dental_tfilm_e10_model.h5"
model_path = os.path.abspath(os.getcwd()) + "/logs/TIMIT_mask_dental_quieter_tfilm_mse_b32_e3_shuffle.h5"
 
# in_dir_lr = "/home/donghyun/Project/IITP/MIR/DB_VCTK/Single/test_lr/mask_dental/"
# in_dir_hr = "/home/donghyun/Project/IITP/MIR/DB_VCTK/Single/test_hr/"
in_dir_lr = "/home/donghyun2/Research/DB_TIMIT/test_lr/anechoic/10_5_44100/dental_quieter/"
in_dir_hr = "/home/donghyun2/Research/DB_TIMIT/test_hr_name/"

# lr_folder = "/home/donghyun/Project/IITP/MIR/DB_VCTK/Single/test_lr/mask_dental/"
# lr_folder = "/home/donghyun2/Research/DB_TIMIT/test_lr/anechoic/10_5_44100/dental_quieter/"
lr_folder = "/home/donghyun2/Research/DB_TIMIT/test_hr_name/"
# lr_folder = "/home/donghyun/Project/IITP/MIR/DB_VCTK/Multispeaker/test_hr/"

# save_folder = "/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/train_dental_multi_tfilm_e100/"
# save_folder = "/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_multi/mask_mixed/tfilm_30/inference2/train2_lpf_notched/"
# save_folder = "/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_multi/mask_dental/afilm_e10/inference2/train2_lpf_notched/"
save_folder = os.path.abspath(os.getcwd()) + '/output/TIMIT_mask_dental_queiter_pr_tfilm_mse_b32_e100/'
# save_folder = os.path.abspath(os.getcwd()) + '/output/TIMIT_hr/'

makedirs(save_folder, exist_ok = True)

def generate_sr_sample(model, crop_length, in_dir_lr, save_path):
    
    sig, sr = librosa.load(], sr=self.sr)

    if len(sig) < self.window:
            sig = pad(sig, self.window)
    batches = int((len(sig) - self.stride) / self.stride)
    sig = sig[0: int(batches * self.stride + self.stride)]
    target = sig.copy()
    return 

if __name__ == '__main__':


    length = x_lr.shape[0]
    # print(length)
    # exit()
    batches = int((length - crop_length / 2) / (crop_length / 2))
    # print(batches)
    # exit()
    x_lr = x_lr[0: int(batches * crop_length / 2 + crop_length / 2)]
    # print(x_lr.shape[0])
    start = time()

    for i in range(batches):
        x_lr_ = x_lr[int(i * crop_length / 2): int((i * crop_length / 2) + crop_length)]
        x_in = np.expand_dims(np.expand_dims(x_lr_, axis=-1), axis=0)
        x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
        pred = model(x_in)
        pred = pred.numpy()
        pred = np.squeeze(np.squeeze(pred))

        if i == 0:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            pred_audio = pred[0: int(crop_length / 2)]
        else:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_overlap = pred_audio_font + pred_audio_end

            pred_audio = np.concatenate((pred_audio, pred_overlap), axis=0)

            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            if i == batches - 1:
                pred_audio = np.concatenate((pred_audio, pred[int(crop_length / 2):]), axis=0)
    # print(pred_audio.shape[0])
    # exit()
    end = time()
    # print('%0.7f' % ((end-start)/10))

    # 4 kHz notch filter
    b, a = signal.iirnotch(4000, 30, fs)
    pred_audio = signal.lfilter(b, a, pred_audio)
     # 2 kHz notch filter
    d, c = signal.iirnotch(2000, 30, fs)
    pred_audio = signal.lfilter(d, c, pred_audio)

    lrpath = save_path.replace('.wav','_lr.wav')
    lr_path = os.path.join(save_path, lrpath)

    hrpath = save_path.replace('.wav','_hr.wav')
    hr_path = os.path.join(save_path, hrpath)
   
    # sf.write(lr_path, x_lr, samplerate=fs)
    # sf.write(hr_path, x_lr, samplerate=fs)
    sf.write(save_path, pred_audio, samplerate=fs)
    
if __name__ == '__main__':

    model = tfilm_net()
    model.load_weights(model_path)
    model.summary()

    # caculate metrics
    # snr, lsd = evaluation(model, crop_length=8192, channel=None,
    #                       in_dir_hr=in_dir_hr, in_dir_lr=in_dir_lr)
    # print("SNR: ", snr, " LSD: ", lsd)

    # generate SR audios from LR audios in 'lr_folder'
    if lr_folder is not None:
        paths = glob(lr_folder + "*.wav")
        paths = natsorted(paths)
        # paths.sort()
        names = os.listdir(lr_folder)
        names = natsorted(names)
        # print(names[0])
        # exit()
        # names.sort()
        num = len(names)
        for i in tqdm(range(num)):
            generate_sr_sample(model, crop_length=8192,
                               in_dir_lr=paths[i], save_path=save_folder + names[i])#[2:])







    if __name__ == '__main__':
  
