# FRN for PLC system

**Modified the baseline PLC system "Masked Frequency Modeling for Improving Packet Loss Concealment in Speech Transmission Systems" - WASPAA 2023**


# 1. Results




# 2. Installation


### Install dependencies

* Our implementation requires the `libsndfile` libraries for the Python packages `soundfile`. On Ubuntu, they can be
  easily installed using `apt-get`:
    ```
    $ apt-get update && apt-get install libsndfile-dev
    ```
* Create a Python 3.8 environment. Conda is recommended:
   ```
   $ conda create -n frn python=3.8
   $ conda activate frn
   ```

* Install the requirements:
    ```
    $ pip install -r requirements.txt 
    ```

# 3. Data preparation



# 4. Run the code

## Configuration

`config.py` is the most important file. Here, you can find all the configurations related to experiment setups,
datasets, models, training, testing, etc. Although the config file has been explained thoroughly, we recommend reading
our paper to fully understand each parameter.

## Training

* Adjust training hyperparameters in `config.py`. We provide the pretrained predictor in `lightning_logs/predictor` as stated in our paper. The FRN model can be trained entirely from scratch and will work as well. In this case, initiate `PLCModel(..., pred_ckpt_path=None)`.

* Run `main.py`:
    ```
    $ python main.py --mode train
    ```
* Each run will create a version in `./lightning_logs`, where the model checkpoint and hyperparameters are saved. In
  case you want to continue training from one of these versions, just set the argument `--version` of the above command
  to your desired version number. For example:
    ```
    # resume from version 0
    $ python main.py --mode train --version 0
    ```
* To monitor the training curves as well as inspect model output visualization, run the tensorboard:
    ```
    $ tensorboard --logdir=./lightning_logs --bind_all
    ```
  ![image.png](https://images.viblo.asia/eb2246f9-2747-43b9-8f78-d6c154144716.png)

## Evaluation

In our paper, we evaluated with 2 masking methods: simulation using Markov Chain and employing real traces in PLC
Challenge.

* Get the blind test set with loss traces:
    ```
    $ wget http://plcchallenge2022pub.blob.core.windows.net/plcchallengearchive/blind.tar.gz
    $ tar -xvf blind.tar.gz -C test_samples
    ```
* Modify `config.py` to change evaluation setup if necessary.
* Run `main.py` with a version number to be evaluated:
    ```
    $ python main.py --mode eval --version 0
    ```
  During the evaluation, several output samples are saved to `CONFIG.LOG.sample_path` for sanity testing.

## Configure a new dataset

Our implementation currently works with the VCTK dataset but can be easily extensible to a new one.

* Firstly, you need to prepare `train.txt` and `test.txt`. See `./data/vctk/train.txt` and `./data/vctk/test.txt` for
  example.
* Secondly, add a new dictionary to `CONFIG.DATA.data_dir`:
    ```
    {
    'root': 'path/to/data/directory',
    'train': 'path/to/train.txt',
    'test': 'path/to/test.txt'
    }
    ```
  **Important:** Make sure each line in `train.txt` and `test.txt` joining with `'root'` is a valid path to its
  corresponding audio file.

# 5. Audio generation

* In order to generate output audios, you need to modify `CONFIG.TEST.in_dir` to your input directory.
* Run `main.py`:
    ```
    python main.py --mode test --version 0
    ```
  The generated audios are saved to `CONFIG.TEST.out_dir`.

  ## ONNX inferencing
  We provide ONNX inferencing scripts and the best ONNX model (converted from the best checkpoint)
  at `lightning_logs/best_model.onnx`.
    * Convert a checkpoint to an ONNX model:
        ```
        python main.py --mode onnx --version 0
        ```
      The converted ONNX model will be saved to `lightning_logs/version_0/checkpoints`.
    * Put test audios in `test_samples` and inference with the converted ONNX model (see `inference_onnx.py` for more
      details):
         ```
        python inference_onnx.py --onnx_path lightning_logs/version_0/frn.onnx
        ```
