import os
import librosa
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

        clean_dataset = os.path.abspath(os.path.expanduser(clean_dataset))
        print(clean_dataset)

        print("Search datasets...")
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files.sort()

        self.length = len(clean_wav_files)
        self.clean_wav_files = clean_wav_files
        self.chunk_size = int(16000)

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        clean_path = self.clean_wav_files[item]
        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
        clean, sr = sf.read(clean_path, dtype="float32")

        assert sr == 16000

        n_frames = (len(clean) - 320) // 160 + 1

        return clean, n_frames, clean_name
