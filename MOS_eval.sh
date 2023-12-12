#!/bin/bash

data_dir='/home/donghyun2/Research/TUNet/TUNet-plc/output/plc-challenge/version_8/PLC/'
clean_dir='/home/donghyun2/Research/TUNet/data/plc-challenge/test/X_CleanReference/'
output_dir='/home/donghyun2/Research/TUNet/TUNet-plc/result/plc-challenge/linux3_ver8_DNSMOS.csv'
output_dir2='/home/donghyun2/Research/TUNet/TUNet-plc/result/plc-challenge/linux3_ver8_PLCMOS.csv'

echo "Evaluate with DNSMOS"
python3 /home/donghyun2/Research/TUNet/TUNet-plc/DNSMOS/dnsmos_local.py \
        --testset_dir $data_dir \
        --csv_path $output_dir && \

echo "Evaluate with PLCMOS" && \
python3 /home/donghyun2/Research/TUNet/TUNet-plc/PLCMOS/plc_mos.py \
        --degraded $data_dir \
        --clean $clean_dir \
        --out-csv $output_dir2 && \
echo "Finished"