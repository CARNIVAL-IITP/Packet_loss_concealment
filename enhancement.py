import argparse
import json
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from utils.stft import STFT
from utils.utils import initialize_config

import os


def main(config, epoch):
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"

    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=16,
    )


    model = initialize_config(config["model"])
    device = torch.device("cpu")
    stft = STFT(
        filter_length=320,
        hop_length=160
    ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()

    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"
    results_dir1 = enhancement_dir / f"primary_{epoch}_epoch"
    results_dir2 = enhancement_dir / f"secondary_{epoch}_epoch"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir1.mkdir(parents=True, exist_ok=True)
    results_dir2.mkdir(parents=True, exist_ok=True)

    for i, (mixture, pitches,  _, names, mixture_path) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]
        print("\tSTFT...")
        mixture_D = stft.transform(mixture)

        #[1, T, F, 2]
        ext = os.path.splitext(name)[-1]
        if ext=='.wav':
            data_dir = os.path.dirname(mixture_path[0])
            txt_dir = data_dir.replace('loss','text')
            txt_name = name.replace('.wav','.txt')
            txt_path = os.path.join(txt_dir, txt_name)

            indexError =[]
            with open(txt_path) as fid:
                lines = fid.readlines()
                for line in lines:
                    indexError.append(int(line.rstrip('\n')))
            nfft_dim = mixture_D.shape[2]
            nFrame = mixture_D.shape[1]

            plc_D = mixture_D # 1,T,161,2예상
            pitches = pitches.unsqueeze(3)
            pitch_cond = torch.cat((pitches,pitches), dim=3) ## 1, 257, 748, 2

            #python error 준걸로 사용할 때
            for index_frame in indexError:
                for index in range(index_frame-2, index_frame+1, 1):
                    if index < 10:
                        pass
                    elif index >= nFrame:
                        pass
                    else:
                        input_fea = mixture_D[:,index-10:index,:,:]
                        input_fea = input_fea.permute(0,3,1,2) # 1,2,10,161예상
                        input_pith = pitch_cond[:,index-10:index,:,:]
                        input_pith = input_pith.permute(0,3,1,2)

                        dataout_plc = model(input_fea, input_pith)
                        dataout_plc = dataout_plc.permute(0,2,3,1) # 1,1,161,2예상
                        plc_D[:,index,:,:] = dataout_plc
                        mixture_D[:,index,:,:] = dataout_plc

            mixture_reshape = mixture_D.permute(0, 3, 2, 1)         # [1, 2, F, T]
            enhanced = mixture_reshape.permute(0, 3, 2, 1) #1,T,F,2 예상
            print("\tEnhancement...")

        enhanced_istft = stft.inverse(enhanced)
        print("enhance time domain", enhanced_istft.shape)                    # torch [1, 80000]
        enhanced_istft = enhanced_istft.detach().cpu().squeeze().numpy()
        sf.write(f"{results_dir}/{name}", enhanced_istft, 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    parser.add_argument("-C", "--config", default="config/enhancement/enhancement_pitch.json", type=str,
                        help="Specify the configuration file for enhancement (*.json).")
    parser.add_argument("-E", "--epoch", default="best",
                        help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    main(config, args.epoch)