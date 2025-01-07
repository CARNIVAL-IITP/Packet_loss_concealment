# FRN for PLC system

**Modified the baseline PLC system "Masked Frequency Modeling for Improving Packet Loss Concealment in Speech Transmission Systems" - WASPAA 2023**

# 0. Contributions

We modified two things.
1. We used complex features in predictor and using FiLM layer to integrate the outputs of encoder and predictor.
2. We used masked frequency modeling method to pretrain the encoder. 


# 1. Installation

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

# 2. Run the code

## Configuration

`config.py` is the most important file. Here, you can find all the configurations related to experiment setups,
datasets, models, training, testing, etc. Although the config file has been explained thoroughly, we recommend reading
our paper to fully understand each parameter.

## Training

* Adjust training hyperparameters in `config.py`.

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
## Evaluation

    ```
* Run `main.py` with a version number to be evaluated:
    ```
    $ python main.py --mode eval --version 0
    ```

# Paper
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10248056

