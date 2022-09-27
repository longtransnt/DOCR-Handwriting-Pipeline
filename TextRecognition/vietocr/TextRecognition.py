import datetime
import json
import os
from venv import create
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from Misc.utils import get_img_list_from_directoty
from TextRecognition.ngram.language_model import LanguageModel, load_data
from TextRecognition.vietocr.vietocr.tool.predictor import Predictor
from TextRecognition.vietocr.vietocr.tool.config import Cfg
from TextRecognition.vietocr.vietocr.model.trainer import Trainer
from TextRecognition.vietocr.vietocr.tool.utils import compute_accuracy
import cv2
import pandas as pd


class TextRecognition(object):

    detector = None
    bigram_lm = None
    trigram_lm = None
    config = Cfg.load_config_from_name('vgg_transformer')

    def __init__(self):

        self.config['weights'] = './TextRecognition/vietocr/weights/transformerocr_new_resorted.pth'
        self.config[
            'vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~°‰µΔ '
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cuda:0'
        self.config['predictor']['beamsearch'] = False

        # self.bigram_lm = self.create_n_gram(n=2)
        # self.trigram_lm = self.create_n_gram(n=3)

    def predict(self, filename, td_output_path, adaptive_output_path, tr_output_path, eval_output_path, is_rerun="true"):

        predict_path = tr_output_path + "/" + filename + "_tr.json"
        eval_path = eval_output_path + "/" + filename + "_eval.json"

        is_predict_exist = os.path.exists(predict_path)
        is_eval_exist = os.path.exists(eval_path)

        eval_info = None
        if (is_eval_exist):
            eval_file = open(eval_path)
            eval_info = json.load(eval_file)
        print(is_predict_exist)
        if (is_predict_exist):
            if (is_rerun == "false"):
                predict_file = open(predict_path)
                predict_info = json.load(predict_file)
                return is_predict_exist, predict_info, is_eval_exist, eval_info
            else:
                os.remove(predict_path)

        adaptive_directory = adaptive_output_path + "/" + filename
        tr_img_list = get_img_list_from_directoty(adaptive_directory)
        tr_filenames = [str(Path(x).stem) for x in tr_img_list]

        coordinate_file = open(td_output_path + "/" + filename +
                               "/coordinates.json")
        coord_file_data = json.load(coordinate_file)

        dashed_line = '=' * 130
        head = f'{"filename":40s}\t' \
            f'{"predicted_string (non-bigram)":25s}\t' \
            f'{"predicted_string (bigram)":25s}\t' \
            f'{"prediction_time"}\t' \

        text_recognition_json_result = []

        print(f'{dashed_line}\n{head}\n{dashed_line}')
        for img, name in zip(tr_img_list, tr_filenames):
            if(img.endswith(".json")):
                continue
            split = name.split('-denoised')
            split_name = split[0]
            cor_dict = list(
                filter(lambda line: line['image_name'].split('.jpg')[0] == split_name, coord_file_data))

            datetime1 = datetime.datetime.now()
            prediction, correction = self.infer(img)
            datetime2 = datetime.datetime.now()
            difference = str(datetime2 - datetime1)

            row_output = f'{name:40s}\t{prediction:25s}' \
                f'\t{correction:25s}' \
                f'\t{difference}'

            print(row_output)
            en = correction.encode("utf8")

            cor_dict[0]["predict"] = en.decode("utf8")

            text_recognition_json_result.append(cor_dict[0])

        jsonpath = Path(predict_path)
        jsonpath.write_text(json.dumps(text_recognition_json_result))

        predict_file = open(predict_path)
        predict_info = json.load(predict_file)
        return is_predict_exist, predict_info, is_eval_exist, eval_info

    def create_n_gram(self, n, laplace=0.1, num=10):
        data_path = Path('./TextRecognition/ngram/data/')
        train, test = load_data(data_path)

        print("Loading {}-gram model...".format(n))
        lm = LanguageModel(train, n, laplace=laplace)
        print("Vocabulary size: {}".format(len(lm.vocab)))

        print("Generating sentences...")
        for sentence, prob in lm.generate_sentences(num):
            print("{} ({:.5f})".format(sentence, prob))

        perplexity = lm.perplexity(test)
        print("Model perplexity: {:.3f}".format(perplexity))
        print("")
        return lm

    def infer(self, im_path, ngram=False):
        img = Image.open(im_path)
        detector = Predictor(self.config)
        next_token = None
        next_prob = None
        prediction = detector.predict(img)

        words = prediction.split()
        correction = ""
        for index, word in enumerate(words):
            word = self.dictionaryCorrection(word)
            correction += " " + word

        return prediction, correction

    def levenshteinDistance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
        # Something something

    def dictionaryCorrection(self, word):
        dictionary = {"đang": 0, "động": 1, "Magnesulfate": 3, "Midazolam": 2, "Arduan": 1,
                      "gồng": 1, "HA": 1, "lần": 1, "Hết": 1, "kiểm": 1, "SpO2": 1, "mmHg": 2, "FiO2": 1, "tỉnh": 1, "nằm": 1}
        distance_resuls = {}
        for d, value in dictionary.items():
            distance = self.levenshteinDistance(word.lower(), d.lower())
            if distance == 0:
                return word
            distance_resuls[d] = distance
        best_key = min(distance_resuls, key=distance_resuls.get)
        if distance_resuls[best_key] <= dictionary[best_key]:
            return best_key
        return word


def evaluation(ground_truths, predicts):
    cer = 1
    wer = 1

    # request_data = request.get_json()

    # ground_truths = request_data["ground_truths"]
    ground_truths = [s.lower() for s in ground_truths]

    # predicts = request_data["predicts"]
    predicts = [s.lower() for s in predicts]

    cer = compute_accuracy(ground_truths, predicts, "cer")
    wer = compute_accuracy(ground_truths, predicts, "per_word")

    return wer, cer
