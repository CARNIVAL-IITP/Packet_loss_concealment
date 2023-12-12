import os, glob
from natsort import natsorted
original_dir = '/home/donghyun2/Research/TUNet/data/plc-challenge/test/blind/lossy_signals/'
print("Original filepath is: ", original_dir)
os.listdir(original_dir)

all_files = glob.glob(original_dir +"*.wav")
all_files = natsorted(all_files)
print("length of files is", len(all_files))
# exit()
# print(all_files)
for i in range(len(all_files)):
    filename = os.path.basename(all_files[i])
    print(filename)
    filename = 'test/blind/lossy_signals/' + filename 
    # print(filename)
    # exit()
    with open('test_noisy.txt', 'a') as f:
        f.write(filename)
        f.write("\n")
        # exit()
exit()
