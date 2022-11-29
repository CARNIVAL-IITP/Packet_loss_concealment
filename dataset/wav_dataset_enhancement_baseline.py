import os

import librosa
import soundfile as sf
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

        mixture, sr = sf.read(mixture_path, dtype="float32")

        assert sr == 16000

        n_frames = (len(mixture) - 320) // 160 + 1

        return mixture, n_frames, name, mixture_path
