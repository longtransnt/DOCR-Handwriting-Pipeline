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
                    # box_res = {}
                    # box_res['box'] = [round(x) for x in bbox[:-1]]
                    # box_res['box_score'] = float(bbox[-1])
                    # box = bbox[:8]
                    if len(bbox) > 9:
                        min_x = min(bbox[0:-1:2])
                        min_y = min(bbox[1:-1:2])
                        max_x = max(bbox[0:-1:2])
                        max_y = max(bbox[1:-1:2])
                        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    
                    path, file_name = os.path.split(filename)
                    file_name,suffix = os.path.splitext(file_name)

                    # Append coordinates of a box to data frame
                    box_coordinates = pd.Series([file_name + "_" + str(count) +".jpg", min_x, min_y, max_x, max_y,filename], index=box_column_names)
                    boxes_coordinates = boxes_coordinates.append(box_coordinates, ignore_index=True)

                    # box_img = crop_img(arr, box)
                    # box_imgs.append(box_img)

                    cropped_images.append(box_imgs)
                    count += 1

        boxes_coordinates.to_csv(output_path + 'boxes_coordinates_'+'_.csv')

            # print("--------------------")
            # boxes_coordinates = boxes_coordinates.astype(int, errors='ignore')
            # print('boxes_coordinates:\n', boxes_coordinates)

            # # Export coordinates of boxes to csv
        




        

    






