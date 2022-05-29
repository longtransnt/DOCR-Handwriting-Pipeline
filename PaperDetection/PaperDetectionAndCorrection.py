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

MODULE_NAME = "PaperDetection"

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class MaskCRNN(object):
    cfg = get_cfg()
    output_path = ""
    output_annotation_path = ""

    def __init__(self, output_path, annotated_output_path):
        self.output_path = output_path
        self.output_annotation_path = annotated_output_path 
        
        self.cfg.merge_from_file(
            "./PaperDetection/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = "./PaperDetection/weights/model_final_0205.pth"  # initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.MAX_ITER = (
            300
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster, and good enough for this toy dataset
        self.cfg.OUTPUT_DIR = './output/'
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        # self.cfg.DATASETS.TEST = ("bills",)
        self.predictor = DefaultPredictor(self.cfg)

    def annotate(self, im, name, data):
        outputs = self.predictor(im)
        temp_im = im.copy()

        pred_masks_list = outputs["instances"].pred_masks.to('cpu').tolist()
        if len(pred_masks_list) > 0:
            # Pred masks for class 0
            pred_masks = pred_masks_list[0]

            # Polygons from masks
            polygons = Mask(pred_masks).polygons()

            # Take the first polygons and find the minimum bounding box (there is only 1 class)
            # This 'box' varriable hold the 4 corners of the predicted text area in our model - Long
            min_rect = cv2.minAreaRect(polygons.points[0])  
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)

            # Sort the array else the order will be fucked
            box = box[box[:, 1].argsort()]
            if (box[0][0] > box[1][0]):
                 box[[0,1]] = box[[1,0]]

            if(box[2][0] < box[3][0]):
                box[[2,3]] = box[[3,2]]
            
            # box = self.alterPredictionBoundingBox(box, 25)
            # print(box)

            v = Visualizer(im[:, :, ::-1],
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                        )
            
            # cv2.drawContours(im,[box],0,(0,0,255),2) #Draw Box of Predicted ROI

            
            roi_paper_img = four_point_transform(im, box)
            stuff = self.cropDarkEdges(roi_paper_img)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            self.convertToAnnotation( name=name, data=data, box=box)
            # Exporting image with prediction contours
            # cv2.imwrite(self.output_annotation_path + "/" + "out_" + name + ".jpg", roi_paper_img)
            return stuff

    def predict(self, im, name, data):
        outputs = self.predictor(im)
        temp_im = im.copy()

        pred_masks_list = outputs["instances"].pred_masks.to('cpu').tolist()
        if len(pred_masks_list) > 0:
            # Pred masks for class 0
            pred_masks = pred_masks_list[0]

            # Polygons from masks
            polygons = Mask(pred_masks).polygons()

            # Take the first polygons and find the minimum bounding box (there is only 1 class)
            # This 'box' varriable hold the 4 corners of the predicted text area in our model - Long
            min_rect = cv2.minAreaRect(polygons.points[0])
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)
         
            v = Visualizer(im[:, :, ::-1],
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                        )

            roi_bill_img = four_point_transform(im, box)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # + ("/" if self.output_path[-1] != '/' else "")
            image_name = self.output_path + "/"  + name + "pd" + ".jpg"
            
            cv2.imwrite(image_name , roi_bill_img)

            return image_name

    def convertToAnnotation(self, name, data, box):
        annotatation = {
            'version': "",
            'flags': {},
            'shapes': [{
                'label': "paper",
                'points': [
                    [
                        float(box[0][0]),
                        float(box[0][1])
                    ],
                    [
                        float(box[1][0]),
                        float(box[1][1])
                    ],
                    [
                        float(box[2][0]),
                        float(box[2][1]),
                    ],
                    [
                        float(box[3][0]),
                        float(box[3][1])
                    ]
                ],
                'group_id': 'null',
                'shape_type': "",
                'flags': {}
            }],
            'imagePath': str(name) + '.jpg',
            'imageData': str(data.decode("utf-8")),
            'imageHeight': float(box[2][1]),
            'imageWidth': float(box[2][0])
        }

        filename = self.output_annotation_path + "/" + name + '.json'
        print("inside path", filename)
        # Using a JSON string
        with open(filename, 'w') as outfile:
            json.dump(annotatation, outfile)

    def alterPredictionBoundingBox(self, box, lenghthenRate = 10):
        yList = [box[0][1], box[1][1], box[2][1], box[3][1]]
        currentBottomYValue = max(yList)
        currentTopYValue = min(yList)
        h = (currentBottomYValue - currentTopYValue)
        adjustedHeight =  h * (lenghthenRate/100)
        box[2][1] =  currentBottomYValue + adjustedHeight
        box[3][1] =  currentBottomYValue + adjustedHeight
        return box

    def cropDarkEdges(self, image):
        y_nonzero, x_nonzero, _ = np.nonzero(image)
        return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    def correctionFeature(self):
        # ROI Moust Drag Code
            roi = cv2.selectROI(im)
            #print(roi)

            if roi == (0,0,0,0):
                print("here")
                roi_bill_img = four_point_transform(im, box)
            else:
                 roi_bill_img = im[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

