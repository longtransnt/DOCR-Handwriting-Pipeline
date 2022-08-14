from venv import create
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from TextRecognition.vietocr.vietocr.tool.config import Cfg
from TextRecognition.vietocr.vietocr.model.trainer import Trainer
import cv2

config = Cfg.load_config_from_name('vgg_transformer')

config[
    'vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~°‰µΔ '

dataset_params = {
    'name': 'resorted_adaptive_1000',
    'data_root': './TextRecognition/vietocr/Sorted/',
    'train_annotation': 'annotation_full.txt',
    'valid_annotation': 'annotation_test_resorted.txt'
}

predictor_params = {
    'beamsearch': False
}
params = {
    'batch_size': 16,
    'print_every': 100,
    'valid_every': 200,
    'iters': 10000,
    'checkpoint': './TextRecognition/vietocr/weights/transformerocr_checkpoint.pth',
    'export': './TextRecognition/vietocr/weights/transformerocr_test.pth',
    'metrics': 100
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['predictor'].update(predictor_params)
config['device'] = 'cuda:0'
config['weights'] = './TextRecognition/vietocr/weights/transformerocr_new_resorted.pth'

trainer = Trainer(config, pretrained=False)

trainer.train()
