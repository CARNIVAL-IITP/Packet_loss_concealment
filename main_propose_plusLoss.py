import argparse
import os

import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader

from config import CONFIG
from dataset_plusLoss import TrainDataset, TestLoader, BlindTestLoader, GenTestLoader, GEN_REAL_TestLoader
from models.frn_propose_plusLoss import PLCModel, OnnxWrapper
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
    # print('checkpoint',checkpoint)

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
        # pretrain_model = PLCModel.load_from_checkpoint('lightning_logs/version_29/checkpoints/frn-epoch=123-val_loss=0.2804.ckpt')

        # pretrained_dict = torch.load('lightning_logs/version_25/checkpoints/frn-epoch=90-val_loss=0.2806.ckpt',map_location='cpu')
        # model_dict = model.state_dict()
        # # print('전부인가', pretrained_dict['state_dict'])
        # pretrained_dict['state_dict'] = {key.replace('encoder.', ''): pretrained_dict['state_dict'].pop(key) for key in pretrained_dict['state_dict'].copy().keys()}
        # # print('바뀌었나', pretrained_dict['state_dict'])
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(pretrained_dict,strict=False)
        # print(PLCModel.encoder)
        # exit()
        # check = model.load_from_checkpoint('lightning_logs/version_29/checkpoints/frn-epoch=123-val_loss=0.2804.ckpt')
        # print(check, type(check))
        # encoder_dict = check.encoder
        # print('org dict',encoder_dict, type(encoder_dict))
        # model.encoder.load_state_dict(encoder_dict)
        # print('model',model)
        # pretrained_dict = torch.load('lightning_logs/version_25/checkpoints/frn-epoch=90-val_loss=0.2806.ckpt')
        # # print(pretrained_dict['state_dict'].keys())
        # # model.load_state_dict(pretrained_dict.encoder)
        # # print('model',model)
        # # model.load_state_dict(pretrained_dict[])
        # model_dict = model.encoder.state_dict()
        # # # print('model 1', model_dict)
        # # print('0',pretrained_dict['state_dict'])
        # for name, param in pretrained_dict['state_dict'].items():
        #     # print('0',name)
        #     name = name.replace('encoder.','')
        #     if name in model_dict:
        #         model_dict.update(pretrained_dict)
        #     # print('1',name)
        # # print('2',pretrained_dict['state_dict'])
        # exit()

        # pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if 'encoder.'+k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(pretrained_dict)
        # print('pre',pretrained_dict)
        # print('model', model_dict)
        # exit()
        # check = {k: v for k, v in check.items() if k in encoder_dict}
        # encoder_dict.update(check)
        # print('new dict',encoder_dict)
        # print('model',model)
        # for name, param in check.encoder.parameters():
        #     print('n,p',name,param)
        # exit()
            # model.encoder

        # model.encoder.load_from_checkpoint('lightning_logs/version_29/checkpoints/frn-epoch=123-val_loss=0.2804.ckpt')
        #     model = PLCModel(train_dataset,
        #                      val_dataset,
        #                      window_size=CONFIG.DATA.window_size,
        #                      enc_layers=CONFIG.MODEL.enc_layers,
        #                      enc_in_dim=CONFIG.MODEL.enc_in_dim,
        #                      enc_dim=CONFIG.MODEL.enc_dim,
        #                      pred_dim=CONFIG.MODEL.pred_dim,
        #                      pred_layers=CONFIG.MODEL.pred_layers)

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
            #testset = GEN_REAL_TestLoader() # plc clean을 generator해서 lossy 파일들 저장한 다음 real로 여겨서 불러와서 eval 찍으려고/eval은 원래 onthefly 방식의 gen에서만 측정하도록 되어있
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer.test(model, test_loader)
            print('Version', args.version)
            masking = CONFIG.DATA.EVAL.masking
            prob = CONFIG.DATA.EVAL.transition_probs[0]
            loss_percent = (1 - prob[0]) / (2 - prob[0] - prob[1]) * 100
            print('Evaluate with real trace' if masking == 'real' else
                  'Evaluate with generated trace with {:.2f}% packet loss'.format(prob))

        elif args.mode == 'gen_test': # gen lossy data 추가하기 위ㅏㅁ. predictor에서 pred 아닌걸로 concat 바꿔줘야
            model.cuda(device=0)
            testset = GenTestLoader()  # WSJ gen으로 predict하는거 저장하려고 추가해널봄. dataset에서 testloader output 첫번쨰꺼로만 받게해야함
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            preds = trainer.predict(model, test_loader, return_predictions=True)
            mkdir_p(CONFIG.TEST.out_dir)
            for idx, path in enumerate(test_loader.dataset.data_list):
                out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                sf.write(out_path, preds[idx], samplerate=CONFIG.DATA.sr, subtype='PCM_16')

        elif args.mode == 'test':
            model.cuda(device=0)
            testset = BlindTestLoader(test_dir=CONFIG.TEST.in_dir) # 오리지널
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
