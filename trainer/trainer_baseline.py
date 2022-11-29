import torch
from pathlib import Path

from trainer.base_trainer_baseline import BaseTrainer
import matplotlib.pyplot as plt

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})

from tqdm import tqdm

class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.root_dir = (Path(config["save_location"]) / config["experiment_name"]).expanduser().absolute()
        self.checkpoints_dir = self.root_dir / "checkpoints"


        print(self.train_dataloader)

    def _train_epoch(self, epoch):
        print('\n')
        
        loss_total = 0.0
        for clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):

            self.optimizer.zero_grad()

            clean = clean.to(self.device)           # time domain

            # 이거 대신 clean을 rir1하고 conv연산하기
            clean_D = self.stft.transform(clean) # filter_size=320 / hop_size=160 [batch, T, F, real/im]
            clean_reshape = clean_D.permute(0, 3, 1, 2) # [B, 2, T, F]

            for i in range(clean_reshape.shape[2]-10):
                if i==0:
                    frame_complex_11unit = clean_reshape[:,:,i:i+10,:]
                    frame_complex_1unit = clean_reshape[:,:,i+10,:]
                else:
                    frame_complex_11unit = torch.cat([frame_complex_11unit, clean_reshape[:,:,i:i+10,:]],dim=0)
                    frame_complex_1unit = torch.cat([frame_complex_1unit, clean_reshape[:,:,i+10,:]],dim=0)

            enhanced_concat = self.model(frame_complex_11unit)

            loss = self.loss_function(enhanced_concat, frame_complex_1unit)
            loss.backward()

            self.optimizer.step()

            loss_total += float(loss)   # training loss

        save_path = "model_epoch%d" % epoch
        torch.save(self.model, self.checkpoints_dir/save_path)
        dataloader_len = len(self.train_dataloader)

        print("train loss = ", loss_total / dataloader_len)

        return loss_total/dataloader_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        mixture_mean = None
        mixture_std = None

        for clean, n_frames_list, _ in tqdm(self.validation_dataloader):

            clean = clean.to(self.device)
            clean_D = self.stft.transform(clean)
            clean_reshape = clean_D.permute(0, 3, 1, 2)     # [B, 2, T, F]

            for i in range(clean_reshape.shape[2]-10):
                if i==0:
                    frame_complex_11unit = clean_reshape[:,:,i:i+10,:]
                    frame_complex_1unit = clean_reshape[:,:,i+10,:]
                else:
                    frame_complex_11unit = torch.cat([frame_complex_11unit, clean_reshape[:,:,i:i+10,:]],dim=0)
                    frame_complex_1unit = torch.cat([frame_complex_1unit, clean_reshape[:,:,i+10,:]],dim=0)

            enhanced_concat = self.model(frame_complex_11unit)

            loss = self.loss_function(enhanced_concat, frame_complex_1unit)
            loss_total += float(loss)  # validation loss


        dataloader_len = len(self.validation_dataloader)
        print("validation loss = ", loss_total / dataloader_len)

        return loss_total / dataloader_len

