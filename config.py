class CONFIG:
    gpus = "0,1,2,3"  # List of gpu devices

    class TRAIN:
        batch_size = 32  # number of audio files per batch
        lr = 1e-4  # learning rate
        epochs = 100 #150  # max training epochs
        workers = 12  # number of dataloader workers
        val_split = 0.1  # validation set proportion
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor

    # Model config
    class MODEL:
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_dim = 512  # dimension of the LSTM in the predictor
        pred_layers = 1  # number of LSTM layers in the predictor

    # Dataset config
    class DATA:
        dataset = 'wsj' #'vctk'  # dataset to use
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'vctk': {'root': 'data/vctk/wav48',
                             'train': "data/vctk/train.txt",
                             'test': "data/vctk/test.txt"},
                    'vctk-0.92-multi': {'root': '/DB/PLC/data/vctk-0.92',
                                        'train': "/DB/PLC/data/vctk-0.92/vctk-0.92_multi_train.txt",
                                        'test': "/DB/PLC/data/vctk-0.92/vctk-0.92_multi_test.txt"},
                    'plc-challenge': {'root': '/DB/plc-challenge',
                                      'train': "/DB/plc-challenge/train_clean.txt",
                                      'train_noisy': "/DB/plc-challenge/train_noisy.txt",
                                      'val': "/DB/plc-challenge/val_clean.txt",
                                      'val_noisy': "/DB/plc-challenge/val_noisy.txt",
                                      'test': "/DB/plc-challenge/test_clean.txt",
                                      'test_noisy': "/DB/plc-challenge/test_noisy.txt",
                                      'test_lossy': "/DB/plc-challenge/test_lossy.txt",
                                      'test_wsj': "/DB/plc-challenge/WSJ/test/wsj_test_clean.txt",},
                    'wsj': {'root': '/DB/plc-challenge/WSJ/test',
                                      'train': "/DB/plc-challenge/WSJ/test/wsj_train_clean.txt",
                                      'val': "/DB/plc-challenge/WSJ/test/wsj_dev_clean.txt",
                                      'test': "/DB/plc-challenge/WSJ/test/wsj_test_clean.txt",
                                      'test_gen': "/DB/plc-challenge/test_lossy.txt",
                                      'test_clean': "/DB/plc-challenge/WSJ/test/wsj_test_clean.txt", }
                    }
#'test_wsj': "/DB/plc-challenge/WSJ/test/wsj_test_clean.txt",
# '/DB/plc-challenge/WSJ/test' #'/DB/plc-challenge
        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 16000  #48000 # audio sampling rate
        audio_chunk_len = 40960 #122880  # size of chunk taken in each audio files
        window_size = 320  #960 # window size of the STFT operation, equivalent to packet size
        stride = 160  #480 # stride of the STFT operation

        class TRAIN:
            packet_sizes = [80, 128, 160, 256, 320, 512]
            #[256, 512, 768, 960, 1024,1536]  # packet sizes for training. All sizes should be divisible by 'audio_chunk_len'
            transition_probs = ((0.9, 0.1), (0.9, 0.5), (0.5, 0.1), (0.5, 0.5))  # list of trainsition probs for Markow Chain #(0.9, 0.5) 16.7 /(0.5, 0.1) 35.7

        class EVAL:
            packet_size = 320  # 20ms
            transition_probs = [(0.9, 0.1)]  # (0.9, 0.1) ~ 10%; (0.8, 0.2) ~ 20%; (0.6, 0.4) ~ 40% // (0.7, 0,1)~25% (0.7,0.3)~30% (0.5, 0.5)~50%
            masking = 'real'  # whether using simulation or real traces from Microsoft to generate masks
            assert masking in ['gen', 'real']
            trace_path = '/DB/plc-challenge/test/blind/lossy_signals' #'/DB/plc-challenge/test/blind/lossy_signals' #'/DB/plc-challenge/WSJ/test/gen/lossy/50%' #  # must be clarified if masking = 'real'

    class LOG:
        log_dir = 'lightning_logs'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.

    class TEST:
        in_dir = '/DB/plc-challenge/test/blind/lossy_signals' #'/DB/plc-challenge/test/blind/lossy_signals' #'/DB/plc-challenge/WSJ/test/gen/lossy/50%'  # path to test audio inputs
        out_dir = '/DB/plc-challenge/test/blind/lossy_signals_output' #'/DB/plc-challenge/test/blind/lossy_signals_output' #'/DB/plc-challenge/WSJ/test/gen/baseline/50%'  # path to generated outputs

        #'/DB/plc-challenge/WSJ/test/gen/proposed_with/10%'
        # '/DB/plc-challenge/test/blind_set_reference/X_CleanReference'
