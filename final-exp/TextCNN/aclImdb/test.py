import os
from tqdm import tqdm
# 网络参考：https://github.com/XqFeng-Josie/TextCNN


# filepath = './train/neg/'
filepath = './train/pos/'

# filepath = './test/'
filename = os.listdir(filepath)

# filename_write = 'rt-polarity.neg'
filename_write = 'rt-polarity.pos'
with open(filename_write,'w') as f_write:
    for f_item in tqdm(filename):
        # print(f_item)
        with open(filepath + f_item, 'r', errors='ignore') as f_read:
            contents = f_read.read()
            f_write.write(contents + '\n')
