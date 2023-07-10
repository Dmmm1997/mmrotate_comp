# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import csv
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np
import sys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', default="/mnt/data/DM_Data/rs_rotatedet_comp/test_fs", help='Image file')
    parser.add_argument('--outcsv', default="work_dirs/rtmdet_test/rotated_rtmdet_l-3x-aug-data40_tta/results_ss_36ep_aug_ttaflip_thr0.1.csv", help='Path to output file')
    parser.add_argument('--sourcecsv', default="work_dirs/rtmdet_test/rotated_rtmdet_l-3x-aug-data40_tta/results_ss_36ep_aug_ttaflip.csv", help='Path to output file')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Fw = open(args.outcsv, "w",newline="")
    writer = csv.writer(Fw)
    txt_element = ["ImageID", "LabelName", 'X1', 'Y1', 'X2','Y2', 'X3', 'Y3','X4', 'Y4','Conf']
    writer.writerow(txt_element)
    with open(args.sourcecsv, 'r') as F:
        reader = csv.reader(F)
        next(reader)
        for row in reader:
            if float(row[-1])>args.score_thr:
                writer.writerow(row)
    Fw.close()

                
    
main()