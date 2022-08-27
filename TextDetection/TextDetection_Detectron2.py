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


class FasterRCNN(object):
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
        # initialize from model zoo
        self.cfg.MODEL.WEIGHTS = "./TextDetection/weights/td_model_final.pth"
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.MAX_ITER = (
            300
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster, and good enough for this toy dataset
        self.cfg.OUTPUT_DIR = './td_d2_output/'
        # 3 classes (data, fig, hazelnut)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        # self.cfg.DATASETS.TEST = ("bills",)
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, original, name, data):
        td_input_img = cv2.imread(name)

        original_path = name[:-7] + ".jpg"

        path, original_file_name_full = os.path.split(original_path)
        original_bare_file_name, suffix = os.path.splitext(
            original_file_name_full)

        if not os.path.exists(self.output_path + "/" + original_bare_file_name):
            os.mkdir(os.path.join(self.output_path,  original_bare_file_name))

        outputs = self.predictor(td_input_img)

        v = Visualizer(original[:, :, ::-1],
                       #    metadata=valid_metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        o = Visualizer(td_input_img[:, :, ::-1],
                       scale=1, instance_mode=ColorMode.IMAGE)

        o = o.draw_instance_predictions(outputs["instances"].to("cpu"))

        visualize_path = self.output_path + "/" + \
            original_bare_file_name
        cv2.imwrite(visualize_path + "/" + "visualize.jpg",
                    v.get_image()[:, :, ::-1])

        cv2.imwrite(visualize_path + "/" + "visualize-normal.jpg",
                    o.get_image()[:, :, ::-1])

        pred_boxes_list = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        box_column_names = ['image_name', 'min_x', 'min_y',
                            'max_x', 'max_y', "original_image_name", "ground_truth"]
        boxes_coordinates = pd.DataFrame(columns=box_column_names)

        i = 0
        for [min_x, min_y, max_x, max_y] in pred_boxes_list:
            min_x, min_y, max_x, max_y = int(min_x), int(
                min_y), int(max_x), int(max_y)
            # print(min_x, min_y, max_x, max_y)
            image_name_suffix = original_bare_file_name + \
                "_td_d2_" + str(i) + ".jpg"
            image_name = self.output_path + "/" + \
                original_bare_file_name + "/" + image_name_suffix

            new_row = pd.DataFrame([image_name_suffix, min_x, min_y, max_x,
                                   max_y, original_file_name_full, ""], index=box_column_names).T
            boxes_coordinates = pd.concat([boxes_coordinates,
                                          new_row])

            # crop_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            crop_img = original[min_y:max_y, min_x:max_x]
            cv2.imwrite(image_name, crop_img)
            i += 1

        boxes_coordinates.to_json(orient="records", path_or_buf=self.output_path + "/" +
                                  original_bare_file_name + "/" + 'coordinates.json')

        return self.output_path + "/" + original_bare_file_name + "/"
