import os
import random
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 clean_dataset,
                 limit=None,
                 offset=0,
                 ):
        """
        Construct train dataset
        Args:
            mixture_dataset (str): mixture dir (wav format files)
            clean_dataset (str): clean dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset
        """

        clean_dataset = os.path.abspath(os.path.expanduser(clean_dataset))
        print(clean_dataset)

        print("Search datasets...")
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files.sort()

        self.length = len(clean_wav_files)
        self.clean_wav_files = clean_wav_files

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        clean_path = self.clean_wav_files[item]
        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
        clean, sr = librosa.load(clean_path, sr=16000)
        # clean, sr = sf.read(clean_path, dtype="float32")
        # def detect_pitch(y, sr):
        #     pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)
        #     # get indexes of the maximum value in each time slice
        #     max_indexes = np.argmax(magnitudes, axis=0)
        #     # get the pitches of the max indexes per time slice
        #     pitches = pitches[max_indexes, range(magnitudes.shape[1])]
        #     return pitches
        S = np.abs(librosa.stft(clean, n_fft=512, hop_length=160, win_length =320, window='hann'))
        pitches, magnitudes = librosa.piptrack(S=S, sr=sr, n_fft=512, hop_length=160) # add for pitch
        # print('check',pitches.shape, magnitudes.shape) # (257, 640) (257, 640)
        shape = np.shape(pitches)
        nb_samples = shape[0]
        nb_windows = shape[1]
        total_pitch=[]
        for i in range(0, nb_windows):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            total_pitch.append(pitch)
            # print('what', total_pitch, type(total_pitch))
            # if pitch==0 and not total_pitch==[]:
            #     # print('hmmm',total_pitch, len(total_pitch))
            #     total_pitch.append(total_pitch[-1])
            # elif pitch > 600 and not total_pitch==[]:
            #     total_pitch.append(total_pitch[-1])
            # else:
            #     total_pitch.append(pitch)
            #     print(type(pitch))
            # if pitch > 600 :
            #     pitch = float(0)
            # total_pitch.append(pitch) # origin
        # print('t', len(total_pitch)) # 640
        pitch_array = np.array(total_pitch) #, dtype= np.float32
        # print('n',pitch_array.shape) # (640,)
        # print(clean_name, pitch_array, np.mean(pitch_array), np.max(pitch_array))
        # exit() pitch_array, pitch_array.shape,

        # max_indexes = np.argmax(magnitudes, axis=0)
        # pitches_scalar = pitches[max_indexes, range(magnitudes.shape[1])]
        # print(magnitudes.shape, pitches_scalar.shape) # (257, 640) (640,)

        """
        if sr != 16000:
            print(sr, clean_path)
        """
        assert sr == 16000

        """
        rand_time = random.randint(0, clean.shape[0]-160000)
        mixture = mixture[rand_time : rand_time+160000]
        clean = clean[rand_time : rand_time+160000]
        """
        n_frames = (len(clean) - 320) // 160 + 1
        # print(clean.shape, pitches.shape, magnitudes.shape, n_frames) #(98303,) (257, 615) (257, 615) 613
        pitches_scalar = np.expand_dims(pitch_array,axis=1)
        # print('squeeze', pitches_scalar.shape) #squeeze (703, 1)
        # exit()
        return clean, pitches_scalar, n_frames, clean_name
