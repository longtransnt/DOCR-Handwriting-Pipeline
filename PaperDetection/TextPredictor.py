from PIL import Image
from imutils.perspective import four_point_transform
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from tuan_utils import minimum_bounding_rectangle


class TextPredictor(object):
    config = Cfg.load_config_from_file('./configs/config.yml')
    detector = None

    def __init__(self):
        self.config['weights'] = './weights/transformerocr.pth'
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cuda:0'
        # self.config['device'] = 'cpu'  # For omega slow invoke
        self.config['predictor']['beamsearch'] = False
        self.detector = Predictor(self.config)

    def predict_mat(self, mat):
        _img = Image.fromarray(mat)
        text, prob = self.detector.predict(_img, return_prob=True)
        return text, prob

    def predict_bill(self, image, polys):
        prediction_texts = []
        rectangles = []
        probs = []

        for poly in polys:
            poly = poly.astype('int32')
            # convert to numpy (for convenience)
            rect_from_poly = minimum_bounding_rectangle(poly)
            warped = four_point_transform(image, rect_from_poly)

            prediction_text, prob = self.predict_mat(warped)
            prediction_text = prediction_text.replace("\f", "")
            if prob > 0.4:
                probs.append(prob)
                prediction_texts.append(prediction_text)
                rectangles.append(rect_from_poly)
        return prediction_texts, rectangles, probs
