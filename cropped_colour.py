import argparse
from unittest.mock import DEFAULT
from PaperDetection import PaperDetectionAndCorrection
from Preprocessing import AmplifyClaheDeblureSauvolaDenoiseConnCompo as Amp
from TextDetection import TextDetection_Detectron2
import sys
import base64
import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
from Misc import constant

unannotated_csv_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + constant.TEXTDETECRION_FOLDER_SUFFIX + '/Batch 1.2/un-annotated.csv'
ouput_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + constant.TEXTDETECRION_FOLDER_SUFFIX + '/Batch 1.2/'
sorted_coordinate_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + constant.TEXTDETECRION_FOLDER_SUFFIX + '/Sorted/coordinate/coordinate_csv'
original_image_path = constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX + constant.TEXTDETECRION_FOLDER_SUFFIX + '/Sorted/original/'
other = './Output/TextDetection/Batch 1.2/un-annotated.csv'
# Call the make_json function
df = pd.read_csv(unannotated_csv_path)
original_list = []
unannotated_list = list(df['file_name'])
for file in unannotated_list:
    original_list.append(file.rpartition('td_d2_')[0])

original_list = list(set(original_list))
i = 0
for each in original_list:
    each_path = sorted_coordinate_path + '/' + each + '_boxes_coordinates_.csv'
    df_each = pd.read_csv(each_path)

    for (index, row) in df_each.iterrows():
        file_name = row['image_name']
        if (unannotated_list.__contains__(file_name)):
            print(file_name)
            min_x, min_y, max_x, max_y = int(row['min_x']), int(row['min_y']), int(row['max_x']), int(row['max_y'])
            print(min_x, min_y, max_x, max_y)

            im = cv2.imread(original_image_path + row['original_image_name'])
            crop_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            crop_img = crop_img[min_y:max_y, min_x:max_x]
            cv2.imwrite(ouput_path+ row['image_name'] , crop_img)
            i+=1
       
        # 
        # # print(min_x, min_y, max_x, max_y)
        # image_name_suffix = name + "td_d2_" + str(i) + ".jpg"
        # image_name = self.output_path + "/"  + image_name_suffix

        # box_coordinates = pd.Series([image_name_suffix, min_x, min_y, max_x, max_y,original_name], index=box_column_names)
        # boxes_coordinates = boxes_coordinates.append(box_coordinates, ignore_index=True)



        # i+=1
print(i)

