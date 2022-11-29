import os

import librosa
import numpy as np
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 mixture_dataset,
                 limit=None,
                 offset=0,
                 ):

        mixture_dataset = os.path.abspath(os.path.expanduser(mixture_dataset))

        assert os.path.exists(mixture_dataset)

        print("Search datasets...")
        mixture_wav_files = librosa.util.find_files(mixture_dataset, ext="wav", limit=None, offset=offset)
        print(f"\t Original length: {len(mixture_wav_files)}")

        self.length = len(mixture_wav_files)
        self.mixture_wav_files = mixture_wav_files

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.mixture_wav_files[item]
        name = os.path.basename(mixture_path)

        mixture, sr = librosa.load(mixture_path, sr=16000)

        S = np.abs(librosa.stft(mixture, n_fft=512, hop_length=160, win_length=320, window='hann'))
        pitches, magnitudes = librosa.piptrack(S=S, sr=sr)  # add for pitch
        shape = np.shape(pitches)
        nb_samples = shape[0]
        nb_windows = shape[1]
        total_pitch = []
        for i in range(0, nb_windows):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            total_pitch.append(pitch)

        pitch_array = np.array(total_pitch)

        assert sr == 16000

        n_frames = (len(mixture) - 320) // 160 + 1
        pitches_scalar = np.expand_dims(pitch_array, axis=1)

        return mixture, pitches_scalar, n_frames, name, mixture_path
