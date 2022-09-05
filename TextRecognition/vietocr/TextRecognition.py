from venv import create
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from TextRecognition.ngram.language_model import LanguageModel, load_data
from TextRecognition.vietocr.vietocr.tool.predictor import Predictor
from TextRecognition.vietocr.vietocr.tool.config import Cfg
from TextRecognition.vietocr.vietocr.model.trainer import Trainer
import cv2


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

        # if(ngram):
        #     for index, word in enumerate(words):
        #         ngram_pred = ""

        #         print("-"*10)
        #         if (index == 0):
        #             ngram_pred += word

        #         print(index, word)

        #         if (next_token is not None and next_prob is not None):
        #             edit_distance = self.levenshteinDistance(
        #                 word, next_token)
        #             print("edit distance between " + word + " and " +
        #                   next_token + " is " + str(edit_distance))

        #         prev = () if self.bigram_lm.n == 1 else tuple(
        #             words[:index+1])

        #         print("prev", prev)
        #         blacklist = [word] + \
        #             (["</s>"] if index < len(words) - 1 else [])

        #         next_token, next_prob = self.bigram_lm._best_candidate(
        #             prev, 0, without=blacklist)

        #         tri_next_token, tri_next_prob = self.trigram_lm._best_candidate(
        #             prev, 0, without=blacklist)

        #         if (next_token != "</s>"):
        #             if (next_prob > 0.26):
        #                 ngram_pred += " " + next_token

        #         print("N-gram result:")
        #         print(" * next token", next_token)
        #         print(" * next prob", next_prob)

        #         print(" * trigram next token", tri_next_token)
        #         print(" * trigram next prob", tri_next_prob)

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
                print(word)
                return word
            distance_resuls[d] = distance
        best_key = min(distance_resuls, key=distance_resuls.get)
        if distance_resuls[best_key] <= dictionary[best_key]:
            print(best_key)
            return best_key
        return word
