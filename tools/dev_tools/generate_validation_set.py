import os
import numpy as np
import shutil

'''
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_trainval.json
'''

'''
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_test.json
'''


train_txt_dir = "/mnt/data/DM_Data/rs_rotatedet_comp/train/labelTxt"

train_image_dir = "/mnt/data/DM_Data/rs_rotatedet_comp/train/images"

val_txt_dir = "/mnt/data/DM_Data/rs_rotatedet_comp/val/labelTxt"

val_image_dir = "/mnt/data/DM_Data/rs_rotatedet_comp/val/images"

val_ratio = 0.1

train_files = os.listdir(train_txt_dir)
np.random.shuffle(train_files)
selected_val_files = train_files[:int(val_ratio*len(train_files))]

for file in selected_val_files:
    name = file.split(".")[0]
    source_file_txt = os.path.join(train_txt_dir,file)
    target_file_txt = os.path.join(val_txt_dir,file)
    source_file_image = os.path.join(train_image_dir,"{}.bmp".format(name))
    target_file_image = os.path.join(val_image_dir,"{}.bmp".format(name))
    shutil.move(source_file_txt, target_file_txt)
    shutil.move(source_file_image, target_file_image)



