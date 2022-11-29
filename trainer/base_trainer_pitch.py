import json
import torch
from pathlib import Path
from utils.utils import prepare_empty_dir, ExecutionTime
from utils.visualization import TensorboardWriter
from utils.stft import STFT


from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, config, resume: bool, model, optimizer, loss_function):
        self.n_gpu = config["n_gpu"]
        self.device = self._prepare_device(self.n_gpu, config["use_cudnn"])

        self.model = model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        self.optimizer = optimizer
        self.loss_function = loss_function

        # Feature
        self.stft = STFT(
            filter_length=320,
            hop_length=160
        ).to(self.device)

        # Trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_interval = config["trainer"]["validation_interval"]
        self.find_max = config["trainer"]["find_max"]
        self.z_score = config["trainer"]["z_score"]

        self.start_epoch = 1  # Not in the config file, will be update if resume is True
        self.best_score = 0.0 if self.find_max else 100  # Not in the config file, will be update in training and if resume is True
        self.root_dir = (Path(config["save_location"]) / config["experiment_name"]).expanduser().absolute()
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume)

        self.viz = TensorboardWriter(self.logs_dir.as_posix())
        self.viz.writer.add_text("Config", json.dumps(config, indent=2, sort_keys=False), global_step=1)
        self.viz.writer.add_text("Description", config["description"], global_step=1)

        if resume: self._resume_checkpoint()

        print("Model, optimizer, parameters and directories initialized.")
        print("Configurations are as follows: ")
        print(json.dumps(config, indent=2, sort_keys=False))

        config_save_path = (self.root_dir / "config.json").as_posix()
        with open(config_save_path, "w") as handle:
            json.dump(config, handle, indent=2, sort_keys=False)
        self._print_networks([self.model])

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    @staticmethod
    def _prepare_device(n_gpu: int, use_cudnn=True):

        use_cpu = False

        if n_gpu == 0:
            use_cpu = True
            print("Using CPU in the experiment.")
        else:
            assert n_gpu <= torch.cuda.device_count(), \
                f"The number of GPUs is {n_gpu}, which is large than the GPUs actually owned ({torch.cuda.device_count()}) in the machine."

            if use_cudnn:
                print("Using CuDNN in the experiment.")
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
            else:
                print("No CuDNN in this experiment.")

        device = torch.device("cpu" if use_cpu else "cuda:0")

        return device

    def _resume_checkpoint(self):

        latest_model_path = self.checkpoints_dir / "latest_model.tar"

        if latest_model_path.exists():
            print(f"Loading the latest model checkpoint in {latest_model_path}.")

            checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

            self.start_epoch = checkpoint["epoch"] + 1
            self.best_score = checkpoint["best_score"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint["model"])

            print(f"Model checkpoint loaded. Training will begin in the {self.start_epoch} epoch.")
        else:
            print(f"{latest_model_path} does not exist, can't load latest model checkpoint.")

    def _set_model_to_train_mode(self):
        self.model.train()

    def _set_model_to_eval_mode(self):
        self.model.eval()

    def _is_best_score(self, score, find_max=True):
        """Check if the current model is the best model"""
        if find_max and score >= self.best_score:
                self.best_score = score
                return True
        elif not find_max and score <= self.best_score:
                self.best_score = score
                return True
        else:
            return False


    @staticmethod
    def _transform_pesq_range(pesq_score):
        """transform [-0.5 ~ 4.5] to [0 ~ 1]"""
        return (pesq_score + 0.5) * 5

    def _save_checkpoint(self, epoch, is_best=False):

        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": None,
            "optimizer": self.optimizer.state_dict(),
        }

        if self.device.type == "cuda" and self.n_gpu > 1:
            # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()

        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        self.model.to(self.device)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            if epoch == 1:
                self.writer = SummaryWriter("./output/tensorboard")

            self._set_model_to_train_mode()
            train_loss = self._train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and epoch % self.save_checkpoint_interval== 0:
                print(f"[{timer.duration()} seconds] Training is over, Saving...")
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration()} seconds] Validation is in progress...")
                self._set_model_to_eval_mode()
                valid_loss = self._validation_epoch(epoch)

                if self._is_best_score(valid_loss, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)

            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("validation_loss", valid_loss, epoch)

            print(f"[{timer.duration()} seconds] End this epoch.")

        self.writer.close()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError
