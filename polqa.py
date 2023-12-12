import os, glob

import numpy as np
from tqdm.auto import tqdm
from natsort import natsorted

from algorithmLib import compute_audio_quality

input_clean_path = '/home/donghyun2/Research/TUNet/TUNet-plc/output/plc-challenge/hr/'
input_enhanced_path = '/home/donghyun2/Research/TUNet/TUNet-plc/output/plc-challenge/lr/'

def evaluate_dataset(input_clean_path, input_enhanced_path):
    results = []

    hr_files = os.listdir(input_clean_path)
    # hr_files.sort()
    hr_files = natsorted(hr_files)
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(input_clean_path + hr_file)

    lr_files = os.listdir(input_enhanced_path)
    # lr_files.sort()
    lr_files = natsorted(lr_files)
    lr_file_list = []
    for lr_file in lr_files:
        lr_file_list.append(input_enhanced_path + lr_file)
  
    # file_num = len(hr_file_list)
    # assert file_num == len(lr_file_list)

    all_files = glob.glob(input_clean_path +"*.wav")
    all_files = natsorted(all_files)
    file_length = len(all_files)
    for i in tqdm(range (800)):
        # x_hr, fs = sf.read(hr_file_list[i])
        # pred, fs = sf.read(lr_file_list[i])
        polqa = compute_audio_quality(metrics='POLQA',testFile=lr_file_list[i],refFile=hr_file_list[i])
        print(polqa)
        exit()


if __name__ == "__main__":

    evaluate_dataset(input_clean_path, input_enhanced_path)
    exit()
    print("PLCMOS:", np.mean(np.array(results["plcmos_v" + str(args.model_ver)])))