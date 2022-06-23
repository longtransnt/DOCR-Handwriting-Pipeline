from pathlib import Path
from re import I
from TextDetection.mmocr.mmocr.utils.ocr import MMOCR
from TextDetection.td_model import textdet_models
# from TextDetection.mmocr.mmocr.datasets.pipelines.crop import crop_img
from pathlib import Path
import pandas as pd
import mmcv
import mmdet
import os
from datetime import date
import datetime
import numpy as np
import cv2
from TextDetection.mmocr.mmocr.utils import check_argument

class TextDetection(object):
    mmocr = None
    input_path = ""
    output_path = ""
    output_annotation_path = ""


    def __init__(self,  output_path):
        self.output_path = output_path 
        self.mmocr = self.initialize_text_detection_model()

    # =============================================================================
    #  Create initial model based on Text Snake
    # =============================================================================
    def initialize_text_detection_model(self):
        for model in textdet_models.keys():
            print("model:", model)
        try:
            mmocr = MMOCR(det="TextSnake", recog=None)
            return mmocr
        except Exception as e:
            print(e)
        
    
    
    def operation(self, img_list):
        self.cropped_and_export_coordinates_to_csv( img_list, self.output_path)
        
    def cropped_and_export_coordinates_to_csv(self, img_list, output_path):
        print("TD out",output_path)
        today = date.today()
        arrays = [mmcv.imread(x) for x in img_list]
        filenames = [str(Path(x).name) for x in img_list]
        cropped_images = []
        print('img_list:', *img_list, sep='\n\t')
        print('filenames:', filenames)
        print("--------------------")

        # # Create a dataframe to export coordiantes of boxes to csv file
        box_column_names = ['image_name', 'min_x', 'min_y', 'max_x', 'max_y', "original_image_name"]
        boxes_coordinates = pd.DataFrame(columns=box_column_names)

        for img_path,filename, arr in zip(img_list,filenames, arrays):
            box_imgs = []
            count = 0
            
            print("path",img_path)
            det_result = self.mmocr.readtext(img_path)

            bboxes_list = [res['boundary_result'] for res in det_result]

            print("filename: ", filename)
            print('bboxes_list size:', len(bboxes_list))
            for bboxes in bboxes_list:
                print('bboxes size:', len(bboxes))
                for bbox in bboxes:
                    if len(bbox) > 9:
                        min_x = min(bbox[0:-1:2])
                        min_y = min(bbox[1:-1:2])
                        max_x = max(bbox[0:-1:2])
                        max_y = max(bbox[1:-1:2])
                        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    
                    path, file_name = os.path.split(filename)
                    file_name,suffix = os.path.splitext(file_name)

                    # Append coordinates of a box to data frame
                    img_name = file_name + "_" + str(count) +".jpg"
                    box_coordinates = pd.Series([img_name, min_x, min_y, max_x, max_y,filename], index=box_column_names)
                    boxes_coordinates = boxes_coordinates.append(box_coordinates, ignore_index=True)

                    box_img = self.crop_img(arr, box)
                    # box_imgs.append(box_img)
                    cv2.imwrite(output_path + "/" + img_name, box_img)
                    count += 1

        boxes_coordinates.to_csv(output_path +"/"+ 'boxes_coordinates_'+'_.csv')


    def crop_img(self, src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
        """Crop text region with their bounding box.

        Args:
            src_img (np.array): The original image.
            box (list[float | int]): Points of quadrangle.
            long_edge_pad_ratio (float): Box pad ratio for long edge
                corresponding to font size.
            short_edge_pad_ratio (float): Box pad ratio for short edge
                corresponding to font size.
        """
        assert check_argument.is_type_list(box, (float, int))
        assert len(box) == 8
        assert 0. <= long_edge_pad_ratio < 1.0
        assert 0. <= short_edge_pad_ratio < 1.0

        h, w = src_img.shape[:2]
        points_x = np.clip(np.array(box[0::2]), 0, w)
        points_y = np.clip(np.array(box[1::2]), 0, h)

        box_width = np.max(points_x) - np.min(points_x)
        box_height = np.max(points_y) - np.min(points_y)
        font_size = min(box_height, box_width)

        if box_height < box_width:
            horizontal_pad = long_edge_pad_ratio * font_size
            vertical_pad = short_edge_pad_ratio * font_size
        else:
            horizontal_pad = short_edge_pad_ratio * font_size
            vertical_pad = long_edge_pad_ratio * font_size

        left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
        top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
        right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
        bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

        dst_img = src_img[top:bottom, left:right]

        return dst_img
            




            

        






