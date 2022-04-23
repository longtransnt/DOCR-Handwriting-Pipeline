import time

import cv2
import torch
from torch.autograd import Variable
from torch.backends import cudnn

from craft import imgproc, craft_utils, file_utils
from craft.craft import CRAFT
from craft.craft_utils import copy_state_dict
from craft.refinenet import RefineNet
import numpy as np


class TextDetector(object):
    net = CRAFT()  # initialize
    refine_net = RefineNet()
    poly = False

    def __init__(self, cuda=True,
                 trained_model='./weights/craft_mlt_25k.pth',
                 refiner_model='./weights/craft_refiner_CTW1500.pth'):
        print('Loading weights from checkpoint for CRAFT')
        if cuda:
            self.net.load_state_dict(copy_state_dict(torch.load(trained_model)))
        else:
            self.net.load_state_dict(copy_state_dict(torch.load(trained_model, map_location='cpu')))

        if cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        if cuda:
            self.refine_net.load_state_dict(copy_state_dict(torch.load(refiner_model)))
            self.refine_net = self.refine_net.cuda()
            self.refine_net = torch.nn.DataParallel(self.refine_net)
            self.refine_net.eval()
            self.poly = True

    def detect(self, img,
               text_threshold=0.2,
               link_threshold=0.3,
               low_text=0.4,
               cuda=True,
               poly=True,
               canvas_size=720,
               mag_ratio=15,
               show_time=True):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if show_time:
            print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
        poly_img = file_utils.get_result(img[:, :, ::-1], polys)
        return boxes, polys, ret_score_text, poly_img


if __name__ == "__main__":
    craft = TextDetector()
    img = cv2.imread(
        '/home/tuan/Documents/mcocr_public_train_test_shared_data/mcocr_val_data/val_images/mcocr_val_145115gozuc.jpg')
    boxes, polys, ret_score_text, im = craft.detect(img)
    cv2.imshow("XD", im)
    cv2.waitKey(0)
