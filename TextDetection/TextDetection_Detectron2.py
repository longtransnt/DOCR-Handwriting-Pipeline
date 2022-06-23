import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from operator import itemgetter
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import numpy as np
from imantics import Mask
from imutils.perspective import four_point_transform
from pathlib import Path
import cv2
from time import time
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import base64
import json
from json import JSONEncoder
import shutil
import pandas as pd


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class TextDetection_FasterRCNN(object):
    cfg = get_cfg()
    output_path = ""
    output_annotation_path = ""

    def __init__(self, output_path, annotated_output_path):
        self.output_path = output_path
        self.output_annotation_path = annotated_output_path 
        
        self.cfg.merge_from_file(
            "./TextDetection/configs/faster_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = "./TextDetection/weights/td_model_final.pth"  # initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.MAX_ITER = (
            300
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster, and good enough for this toy dataset
        self.cfg.OUTPUT_DIR = './td_d2_output/'
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        # self.cfg.DATASETS.TEST = ("bills",)
        self.predictor = DefaultPredictor(self.cfg)
    
    def predict(self, im, name, data):
        original_name = name[:-3] + ".jpg"
        name = name[:-5]
        outputs = self.predictor(im)
        pred_boxes_list = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        box_column_names = ['image_name', 'min_x', 'min_y', 'max_x', 'max_y', "original_image_name"]
        boxes_coordinates = pd.DataFrame(columns=box_column_names)

        i = 0
        for [min_x, min_y, max_x, max_y] in pred_boxes_list:
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            # print(min_x, min_y, max_x, max_y)
            image_name_suffix = name + "td_d2_" + str(i) + ".jpg"
            image_name = self.output_path + "/"  + image_name_suffix

            box_coordinates = pd.Series([image_name_suffix, min_x, min_y, max_x, max_y,original_name], index=box_column_names)
            boxes_coordinates = boxes_coordinates.append(box_coordinates, ignore_index=True)


            crop_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            crop_img = crop_img[min_y:max_y, min_x:max_x]
            cv2.imwrite(image_name , crop_img)
            i+=1

        boxes_coordinates.to_csv(self.output_annotation_path +"/"+ name +'_boxes_coordinates'+'_.csv')




