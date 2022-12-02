import argparse
import json
import os
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
parser = argparse.ArgumentParser(description='EHNET')
parser.add_argument("-C", "--config", default="config/train/train.json", type=str,
                    help="Specify the configuration file for training (*.json).")
parser.add_argument('-D', '--device', default=None, type=str,
                    help="Specify the GPU visible in the experiment, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true",
                    help="Whether to resume training from a recent breakpoint.")
args = parser.parse_args()

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer_pitch import Trainer
from utils.utils import initialize_config
from torch.nn.utils.rnn import pad_sequence

from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(config, resume):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    def pad_to_longest(batch):
        clean_list = []
        pitch_list = []
        names = []
        n_frames_list = []

        for clean, pitches, n_frames, name in batch:
            clean_list.append(torch.tensor(clean).reshape(-1, 1))
            pitch_list.append(torch.tensor(pitches))
            n_frames_list.append(n_frames)
            names.append(name)

        clean_list = pad_sequence(clean_list).squeeze(2).permute(1, 0)
        pitch_list = pad_sequence(pitch_list).permute(1,0,2)

        return clean_list, pitch_list, n_frames_list, names

    train_dataset = initialize_config(config["train_dataset"])

    train_data_loader = DataLoader(
        shuffle=config["train_dataloader"]["shuffle"],
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        collate_fn=pad_to_longest,
        drop_last=True
    )

    validation_dataset = initialize_config(config["validation_dataset"])
    valid_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"],
        collate_fn=pad_to_longest,
        shuffle=config["validation_dataloader"]["shuffle"]

    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        amsgrad=True
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_dataloader=train_data_loader,
        validation_dataloader=valid_data_loader
    )

    trainer.train()


if __name__ == '__main__':
    # load config file
    config = json.load(open(args.config))
    now = datetime.now()
    config["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]+now.strftime('_%y%m%d_%H%M')

    main(config, resume=args.resume)


