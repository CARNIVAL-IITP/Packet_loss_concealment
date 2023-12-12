import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from librosa.display import waveshow
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec


# ## Window filepath
# input_original_path = '/home/utahboy3502/1007MIR/MIR/'
# input_unmasked_path = '/home/utahboy3502/1007MIR/MIR/output/'
# input_masked_path = '/home/utahboy3502/1007MIR/MIR/output/masked/'

# wav_original_name = 'test1.wav'
# unmasked_wav_name = 'unmasked_output.wav'
# masked_wav_name = 'kf99_output.wav'

# outputpath = '/home/utahboy3502/1007MIR/MIR/output/'

input_original_path = '/home/donghyun/Research/TUNet/test/version_7/sample_1/'
input_unmasked_path = '/home/donghyun/Research/TUNet/test/version_7/sample_1/'
input_masked_path = '/home/donghyun/Research/TUNet/test/version_7/sample_1/'
input_masked_path2 = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/tfilm_e50/'

## Linux filepath
# input_original_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/hr_inference2/'
# input_unmasked_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/lr_inference2/'
# input_masked_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/tfilm_mae_e100/inference2/train2_lpf_notched/'
# input_masked_path2 = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/tfilm_e50/'

# wav_original_name = 'p225_355.wav'
# unmasked_wav_name = 'p225_355.wav' 
# masked_wav_name = 'p225_355.wav' 
# masked_wav_name2 = '0.2.pr.wav'

# input_original_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_multi/mask_dental/hr_inference2/'
# input_unmasked_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_multi/mask_dental/lr_inference2/'
# input_masked_path = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_multi/mask_dental/tfilm_e100/inference2/train2_lpf_notched/'
# input_masked_path2 = '/home/donghyun/Research/audio-super-res/data/vctk/VCTK-Corpus/test_single/mask_dental/tfilm_e50/'

wav_original_name = 'high_rate.wav'
unmasked_wav_name = 'low_rate.wav' 
masked_wav_name = 'recon.wav' 
# masked_wav_name2 = '0.2.pr.wav'

outputpath = '/home/utahboy3502/Progress/Project/IITP/MIR/output/1013/'

sr = 16000
hop_length = 64
n_fft = 1024

wav_original = input_original_path + wav_original_name
wav_unmasked = input_unmasked_path + unmasked_wav_name
wav_masked = input_masked_path + masked_wav_name
# wav_masked2 = input_masked_path2 + masked_wav_name2

def plot_waveform_to_numpy(self, y, y_low, y_recon, step):
    name_list = ['y', 'y_low', 'y_recon']
    fig = plt.figure(figsize=(9, 15))
    fig.suptitle(f'Epoch_{step}')
    for i, yy in enumerate([y, y_low, y_recon]):
        ax = plt.subplot(3, 1, i + 1)
        ax.set_title(name_list[i])
        waveshow(yy.numpy(), self.sr)
        # plt.imshow(rosa.amplitude_to_db(self.stftmag(yy).numpy(),
        #                                 ref=np.max, top_db=80.),
        #            # vmin = -20,
        #            vmax=0.,
        #            aspect='auto',
        #            origin='lower',
        #            interpolation='none')
        # plt.colorbar()
        # plt.xlabel('Frames')
        # plt.ylabel('Channels')
        # plt.tight_layout()

    fig.canvas.draw()
    data = self.fig2np(fig)

    plt.close()
    return data

    
def waveform(self, y, y_low, y_recon, epoch):
    y, y_low, y_recon = y.detach().cpu(), y_low.detach().cpu(), y_recon.detach().cpu()
    spec_img = self.plot_waveform_to_numpy(y, y_low, y_recon, epoch)
    self.experiment.add_image(path.join(self.save_dir, 'result'),
                                spec_img,
                                epoch,
                                dataformats='HWC')
    self.experiment.flush()
    return


## plot 3 waveform and spectrogram each
def stft1():
    fig = plt.figure(figsize=(12, 6))

    ##original
    s1_w = fig.add_subplot(2, 3, 1)
    y, sr = librosa.load(wav_original, sr=16000)
    librosa.display.waveplot(y, sr=sr)
    plt.title('Original')

    s1 = fig.add_subplot(2, 3, 4)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(stft)))

    ##unmasked
    s2_w = fig.add_subplot(2, 3, 2)
    y, sr = librosa.load(wav_unmasked, sr=16000)
    librosa.display.waveplot(y, sr=sr)
    plt.title('Unmasked')

    s2 = fig.add_subplot(2, 3, 5)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(stft)))

    ##masked
    s3_w = fig.add_subplot(2, 3, 3)
    y, sr = librosa.load(wav_masked, sr=16000)
    librosa.display.waveplot(y, sr=sr)
    plt.title('Masked')

    s3 = fig.add_subplot(2, 3, 6)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(stft)))

    plt.tight_layout()
    # plt.savefig(outputpath + 'output_stft_plot_.png')
    plt.show()

## plot waveform top, 3 spectrogram bottom
def stft2():
    fig = plt.figure(constrained_layout = True)

    gs = gridspec.GridSpec(2, 3, figure = fig)
    ax = fig.add_subplot(gs[0, :])
    y, sr = librosa.load(wav_original, sr=16000)
    librosa.display.waveplot(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Waveform')

    ax2 = fig.add_subplot(gs[1,0])
    y, sr = librosa.load(wav_original, sr=16000)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original Spectrogram')

    ax3 = fig.add_subplot(gs[1,1])
    y, sr = librosa.load(wav_unmasked, sr=16000)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('KF-99 Mask Spectrogram')

    ax4 = fig.add_subplot(gs[1,2])
    y, sr = librosa.load(wav_masked, sr=16000)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Dental Mask Spectrogram')

    # plt.savefig('dental_fabric_stft.jpg')
    plt.show()



def stft3():
    fig = plt.figure(figsize=(12, 6))

    s1_w = fig.add_subplot(1, 3, 1)
    
    ref, sr = librosa.load(wav_original, sr=16000)
    stft = librosa.stft(y=ref, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original')
    
    s2_w = fig.add_subplot(1, 3, 2)
   
    y, sr = librosa.load(wav_unmasked, sr=sr)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('LPF')
    
    s3_w = fig.add_subplot(1, 3, 3)
    
    deg, sr = librosa.load(wav_masked, sr=16000)
    stft = librosa.stft(y=deg, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Enhanced')

    # s3_w = fig.add_subplot(1, 4, 4)
    
    # deg, sr = librosa.load(wav_masked2, sr=16000)
    # stft = librosa.stft(y=deg, n_fft=n_fft, hop_length=hop_length)
    # magnitude = np.abs(stft)
    # log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    # librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Enhanced epoch 50')

    # score = pesq(ref, deg, sr)
    # print("PESQ Score is: ", score)

    # plt.colorbar(format='%+2.0f dB');
    plt.tight_layout()
    #plt.savefig('mel-spectrogram.png')
    plt.show()
    # plt.savefig('Mel-Spectrogram.png', dpi = 200)
    
    #waveplot
    #librosa.display.waveplot(y, sr=sr)

input_path = '/home/donghyun2/Research/py_utils/demo/230327/'
# ./sample_audio_input/demo_0314/'
# outputpath = '/home/donghyun2/Research/py_utils/deom

# wav_original_name = 'hr_p225_356.wav'
# wav_LPF_name = 'lr_p225_356.wav'
# wav_DNNbase_name = 'e10_p225_356.wav'
# wav_DNNupdate_name = 'e10_notched_p225_356.wav'
# wav_tfilm_name = 'e50_notched_p225_356.wav'
# wav_afilm_name = 'e100_notched_p225_356.wav'

# wav_original_name = '0.4.lr.wav'
# wav_LPF_name = '0.8.lr.wav' 
# wav_DNNbase_name = '0.4.tfilm.wav'
# wav_DNNupdate_name = '0.8.tfilm.wav'
# wav_tfilm_name = '0.4.afilm.wav'
# wav_afilm_name = '0.8.afilm.wav'

wav_original_name = 'p360_001.wav'
wav_LPF_name = 'SA1_lr.wav'
wav_DNNbase_name = 'SA1_ver9.wav' 
wav_DNNupdate_name = 'SA1_ver102.wav'
wav_tfilm_name = 'SA1_ver103.wav'
wav_afilm_name = 'SA1_ver107.wav'


sr = 16000
hop_length = 64
n_fft = 1024

def stft4():
    wav_original = input_path + wav_original_name
    wav_LPF = input_path + wav_LPF_name
    wav_DNNbase = input_path + wav_DNNbase_name
    wav_DNNupdate = input_path + wav_DNNupdate_name
    wav_tfilm = input_path + wav_tfilm_name
    wav_afilm = input_path + wav_afilm_name
    
    fig = plt.figure(figsize=(12, 6))

    ##original
    s1 = fig.add_subplot(2, 3, 1)
    y1, sr = librosa.load(wav_original, sr=16000)
    # librosa.display.waveplot(y, sr=sr)
    stft = librosa.stft(y=y1, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y1) / sr, np.shape(stft)))
    plt.title('Original')

    ## DENTAL
    s2 = fig.add_subplot(2, 3, 4)
    y2, sr = librosa.load(wav_LPF, sr=16000)
    stft = librosa.stft(y=y2, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y2) / sr, np.shape(stft)))
    plt.title('Low pass filtered')

    ## KF99
    s3 = fig.add_subplot(2, 3, 2)
    y3, sr = librosa.load(wav_DNNbase, sr=16000)
    # librosa.display.waveplot(y3, sr=sr)
    stft = librosa.stft(y=y3, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y3) / sr, np.shape(stft)))
    plt.title('TUNet_baseline')

    ## FABRIC
    s4 = fig.add_subplot(2, 3, 5)
    y4, sr = librosa.load(wav_DNNupdate, sr=16000)
    # librosa.display.waveplot(y3, sr=sr)
    stft = librosa.stft(y=y4, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y4) / sr, np.shape(stft)))
    plt.title('TUNet_CTRL: EWC')

    ## KF99
    s5 = fig.add_subplot(2, 3, 3)
    y5, sr = librosa.load(wav_tfilm, sr=16000)
    # librosa.display.waveplot(y, sr=sr)
    stft = librosa.stft(y=y5, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y5) / sr, np.shape(stft)))
    plt.title('TUNet MSM with CTRL: EMA')

    ## FABRIC
    s6 = fig.add_subplot(2, 3, 6)
    y6, sr = librosa.load(wav_afilm, sr=16000)
    # librosa.display.waveplot(y3, sr=sr)
    stft = librosa.stft(y=y6, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    print("Wave length: {}, Mel_S shape:{}".format(len(y6) / sr, np.shape(stft)))
    plt.title('TUNet MSM with CTRL: EWC checkboard')

    plt.tight_layout()
    # plt.savefig(outputpath + 'masked.png')
    plt.show()

# stft1()
# stft2()
# stft3()
stft4()