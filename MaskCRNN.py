import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import numpy as np
from imantics import Mask
from imutils.perspective import four_point_transform
from pathlib import Path

class MaskCRNN(object):
    cfg = get_cfg()
    
    output_path = "/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/med_records_det_output"

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
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "mask_crnn.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        self.cfg.DATASETS.TEST = ("bills",)
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, im, name):
        outputs = self.predictor(im)

        # Pred classes
        # print(outputs["instances"].pred_classes)

        # Pred boxes
        # print(outputs["instances"].pred_boxes)

        # Pred masks
        # print(outputs["instances"].pred_masks)

        pred_masks_list = outputs["instances"].pred_masks.to('cpu').tolist()
        if len(pred_masks_list) > 0:
            # Pred masks for class 0
            pred_masks = pred_masks_list[0]

            # Polygons from masks
            polygons = Mask(pred_masks).polygons()
            # print(polygons.points[0])

            # Take the first polygons and find the minimum bounding box (there is only 1 class)
            min_rect = cv2.minAreaRect(polygons.points[0])
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)

            # Try print the box?
            # print(box)

            v = Visualizer(im[:, :, ::-1],
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                        )

            roi_bill_img = four_point_transform(im, box)

            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(self.output_path + ("/" if self.output_path[-1] != '/' else "") + "out_original_" + name + ".jpg", roi_bill_img)

            # cv2.imwrite("tuan.png", roi_bill_img)
            # cv2.imshow("haha", v.get_image()[:, :, ::-1])

            # Show ROI of image
            # cv2.imshow("haha", roi_bill_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return v.get_image()[:, :, ::-1], roi_bill_img


if __name__ == "__main__":
    # Test
    imgs_dir = [
        # "/mnt/c/Users/Tuan/01NVe/RAW_DATA/2q.000382(25).jpg",
        # "/mnt/c/Users/Tuan/01NVe/RAW_DATA/21,.001477(1).jpg"
        # r"/mnt/c/Users/Tuan/01NVe/RAW_DATA/21,.001477(5).jpg"
        # r"/mnt/c/Users/Tuan/01NVe/RAW_DATA/2q.000382 (15).jpg"
        # r"/mnt/c/Users/Tuan/01NVe/RAW_DATA/2021_11_19 11_51 Office Lens (12).jpg"
        # r"/mnt/c/Users/Tuan/01NVe/RAW_DATA/21.011023A (3).jpg"
        # r"/mnt/c/Users/Tuan/Documents/mcocr_public_train_test_shared_data/mcocr_train_data/train_images/mcocr_public_145013acjke.jpg"
        r"/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/med_records_det/Different",
        # r"/mnt/c/Users/antan/Desktop/OENG1183/Handwriting Recognition/bill_demo/med_records_det/21.000440 (12).jpg"
    ]

    img_list = []

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

    print(filenames)

    for img, name in zip(img_list, filenames):
        # im = cv2.imread()
        im = cv2.imread(img)
        mask = MaskCRNN()
        mask.predict(im=im, name=name)
