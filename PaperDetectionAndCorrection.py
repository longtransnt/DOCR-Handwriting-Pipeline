import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from operator import itemgetter
import cv2
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

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class MaskCRNN(object):
    cfg = get_cfg()
    output_path = "/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/med_records_det_output"
    output_annotation_path = "/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/Subset2/"

    def __init__(self):
        self.cfg.merge_from_file(
            "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.MAX_ITER = (
            300
        )  # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster, and good enough for this toy dataset
        self.cfg.OUTPUT_DIR = './weights/'
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "maskrcnn-100-train.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        self.cfg.DATASETS.TEST = ("bills",)
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, im, name, data):
        outputs = self.predictor(im)
        temp_im = im.copy()
        # Pred classes
        #print(outputs["instances"].pred_classes)

        # Pred boxes
        #print(outputs["instances"].pred_boxes)

        # Pred masks
        #print(outputs["instances"].pred_masks)

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
            print(box)

            # Sort the array else the order will be fucked
            box = box[box[:, 1].argsort()]
            if (box[0][0] > box[1][0]):
                 box[[0,1]] = box[[1,0]]

            if(box[2][0] < box[3][0]):
                box[[2,3]] = box[[3,2]]
            
            # print("box after sort")
            print(box)
            # Try print the box before and after alteration?
            # print("box after box padding")
            box = self.alterPredictionBoundingBox(box, 25)
            # print(box)

            v = Visualizer(im[:, :, ::-1],
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                        )
            
            cv2.drawContours(im,[box],0,(0,0,255),2) #Draw Box of Predicted ROI

            # # Idea, use mouse event to mark the 4 new corners of the box variable, then update it, and display it on the cv2 windows
            # roi = cv2.selectROI(temp_im)
            # # If ROI different from box coordinates, register new annotate coordinate
            # if (int(roi[0]) != 0 or int(roi[1]) != 0 or int(roi[2]) != 0 or int(roi[3]) != 0):
            #     print(roi)
            #     # Assign it to Box
            #     box[0][0] = int(roi[0])
            #     box[0][1] = int(roi[1])
            #     box[1][0] = int(roi[2])
            #     box[1][1] = int(roi[1])
            #     box[2][0] = int(roi[2])
            #     box[2][1] = int(roi[3])
            #     box[3][0] = int(roi[1])
            #     box[3][1] = int(roi[3])
            #     print("Will assign ROI to box")

            #print(box)
            #Show ROI of image
            # cv2.imshow("Output Images", im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            roi_bill_img = four_point_transform(im, box)
            stuff = self.cropDarkEdges(roi_bill_img)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            self.convertToAnnotation(image=im, name=name, data=data, box=box)
            # Exporting image with prediction contours
            cv2.imwrite(self.output_path + ("/" if self.output_path[-1] != '/Prediction/' else "") + "out_" + name + ".jpg", im)

            # cv2.imwrite("tuan.png", roi_bill_img)
            # cv2.namedWindow("Prediction with ROI", cv2.WINDOW_NORMAL)
            # cv2.imshow("Prediction with ROI", roi_bill_img)

            return v.get_image()[:, :, ::-1], stuff

    def convertToAnnotation(self, image, name, data, box):
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

        filename = self.output_annotation_path + name + '.json'
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

    # def cropDarkEdges(self, img):
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #     contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #     cnt = contours[0]
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     crop = img[y:y+h,x:x+w]
    #     cv2.imwrite('sofwinres.png',crop)
    #     return crop

    def correctionFeature(self):
        # ROI Moust Drag Code
            roi = cv2.selectROI(im)
            #print(roi)

            if roi == (0,0,0,0):
                print("here")
                roi_bill_img = four_point_transform(im, box)
            else:
                 roi_bill_img = im[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

if __name__ == "__main__":
    # Test
    imgs_dir = [
         r"/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/Subset2",
    ]

    img_list = []

    # Loading Image List
    for img in imgs_dir:
        # Create a list of the images
        if isinstance(img, str):
            img_path = Path(img)
            
            if img_path.is_dir():
                img_list += [str(x) for x in img_path.glob('*')]
            else:
                img_list += [str(img_path)]
        elif isinstance(img, np.ndarray):
            img_list += [img]

    filenames = [str(Path(x).stem) for x in img_list]

    # 
    for img, name in zip(img_list, filenames):
        # im = cv2.imread()
        im = cv2.imread(img)
        mask = MaskCRNN()
        print(img)
        # # Encode the image as Base64
        with open(img, "rb") as img_file:
            data = base64.b64encode(img_file.read())

        mask.predict(im=im, name=name, data=data)
    
    # # Move from one desitation to another
    # # To be used (?)
    # source_folder = r"/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/Subset2/Annotate"
    # destination_folder = r"/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/Subset2/"

    # # Move all files
    # for file_name in os.listdir(source_folder):
    #     # construct full file path
    #     source = source_folder + file_name
    #     destination = destination_folder + file_name
    #     # move only files
    #     if os.path.isfile(source):
    #         shutil.move(source, destination)
    #         print('Moved:', file_name)
