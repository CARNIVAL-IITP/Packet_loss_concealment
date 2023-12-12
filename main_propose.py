import argparse
import os

import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import TrainDataset, TestLoader, BlindTestLoader, GenTestLoader, GEN_REAL_TestLoader
from models.frn_propose import PLCModel, OnnxWrapper
from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import mkdir_p

parser = argparse.ArgumentParser()

parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='training or testing mode')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train', 'eval', 'test', 'onnx','gen_test'], "--mode should be 'train', 'eval', 'test' or 'onnx'"


def resume(train_dataset, val_dataset, version):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name
    checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=True,
                                               hparams_file=config_path,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               window_size=CONFIG.DATA.window_size,
                                               pred_ckpt_path=None)
    return checkpoint


def train():
    train_dataset = TrainDataset('train')
    val_dataset = TrainDataset('val')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='frn-{epoch:02d}-{val_loss:.4f}', save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)
    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    else:
        model = PLCModel(train_dataset,
                         val_dataset,
                         window_size=CONFIG.DATA.window_size,
                         enc_layers=CONFIG.MODEL.enc_layers,
                         enc_in_dim=CONFIG.MODEL.enc_in_dim,
                         enc_dim=CONFIG.MODEL.enc_dim,
                         pred_dim=CONFIG.MODEL.pred_dim,
                         pred_layers=CONFIG.MODEL.pred_layers,
                         pred_ckpt_path=None)
      
    trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         gpus=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator="gpu" if len(gpus) > 1 else None,
                         callbacks=[checkpoint_callback]
                         )

    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    trainer.fit(model)


def to_onnx(model, onnx_path):
    model.eval()

    model = OnnxWrapper(model)

    torch.onnx.export(model,
                      model.sample,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      input_names=model.input_names,
                      output_names=model.output_names,
                      do_constant_folding=True,
                      verbose=False)


if __name__ == '__main__':

    if args.mode == 'train':
        train()
    else:
        model = resume(None, None, args.version)
        print(model.hparams)
        print(summarize(model))

        model.eval()
        model.freeze()
        if args.mode == 'eval':
            model.cuda(device=0)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            testset = TestLoader() # original
            #testset = GEN_REAL_TestLoader() 
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer.test(model, test_loader)
            print('Version', args.version)
            masking = CONFIG.DATA.EVAL.masking
            prob = CONFIG.DATA.EVAL.transition_probs[0]
            loss_percent = (1 - prob[0]) / (2 - prob[0] - prob[1]) * 100
            print('Evaluate with real trace' if masking == 'real' else
                  'Evaluate with generated trace with {:.2f}% packet loss'.format(prob))

        elif args.mode == 'gen_test': 
            model.cuda(device=0)
            testset = GenTestLoader()  
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            preds = trainer.predict(model, test_loader, return_predictions=True)
            mkdir_p(CONFIG.TEST.out_dir)
            for idx, path in enumerate(test_loader.dataset.data_list):
                out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                sf.write(out_path, preds[idx], samplerate=CONFIG.DATA.sr, subtype='PCM_16')

        elif args.mode == 'test':
            model.cuda(device=0)
            testset = BlindTestLoader(test_dir=CONFIG.TEST.in_dir) 
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            preds = trainer.predict(model, test_loader, return_predictions=True)
            mkdir_p(CONFIG.TEST.out_dir)
            for idx, path in enumerate(test_loader.dataset.data_list):
                out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                sf.write(out_path, preds[idx], samplerate=CONFIG.DATA.sr, subtype='PCM_16')

        else:
            onnx_path = 'lightning_logs/0WSJ/version_{}/checkpoints/frn.onnx'.format(str(args.version))
            to_onnx(model, onnx_path)
            print('ONNX model saved to', onnx_path)
