import argparse
import shutil
from unittest.mock import DEFAULT
import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from Misc import constant


unannotated_csv_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + \
    constant.TEXTDETECTION_FOLDER_SUFFIX + '/Batch 1.2/un-annotated.csv'
ouput_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + \
    constant.TEXTDETECTION_FOLDER_SUFFIX + '/Batch 1.2/'
sorted_coordinate_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + \
    constant.TEXTDETECTION_FOLDER_SUFFIX + '/Batch 1.2/coords_csv/'
original_image_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + \
    constant.TEXTDETECTION_FOLDER_SUFFIX + '/Sorted/original/'
other = './Output/TextDetection/Batch 1.2/un-annotated.csv'


def cropp_colours():
    # Call the make_json function
    # df = pd.read_csv(unannotated_csv_path)
    # original_list = []
    # unannotated_list = list(df['file_name'])
    # for file in unannotated_list:
    #     original_list.append(file.rpartition('td_d2_')[0])

    # original_list = list(set(original_list))
    # print(original_list)

    coordinate_csv_file_list = [f for f in listdir(
        sorted_coordinate_path) if isfile(join(sorted_coordinate_path, f))]
    print(coordinate_csv_file_list)
    i = 0
    for each in coordinate_csv_file_list:
        each_path = sorted_coordinate_path + '/' + each
        df_each = pd.read_csv(each_path)
        # shutil.move(each_path, ouput_path)

        for (index, row) in df_each.iterrows():
            file_name = row['image_name']
            # if (unannotated_list.__contains__(file_name)):
            print(file_name)
            min_x, min_y, max_x, max_y = int(row['min_x']), int(
                row['min_y']), int(row['max_x']), int(row['max_y'])
            print(original_image_path + row['original_image_name'])

            im = cv2.imread(original_image_path + row['original_image_name'])
            # crop_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            crop_img = im[min_y:max_y, min_x:max_x]
            cv2.imwrite(ouput_path + row['image_name'], crop_img)
            i += 1

    print(i)


def write_filename_txt():
    arr = os.listdir(ouput_path)

    for filename in arr:
        if ".jpg" in filename:
            print(filename)


if __name__ == '__main__':
    cropp_colours()
