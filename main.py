import argparse
from operator import contains
from unittest import skip
from Misc.utils import get_img_list_from_directoty
from PaperDetection import PaperDetection
from Preprocessing import AmplifyClaheDeblureSauvolaDenoiseConnCompo as Amp
from TextDetection import TextDetection_Detectron2
from TextDetection_PostProcesscing.AdaptivePreprocesscing import AdaptiveProcessing, applyAdaptivePreprocesscingStep, applyAdaptivePreprocesscingManualStep, denoise
from TextRecognition.vietocr.TextRecognition import TextRecognition, evaluation
from TextRecognition.vietocr.vietocr.tool.utils import compute_accuracy
from datetime import datetime
import time
import sys
import base64
import json
import cv2
import numpy as np
import datetime
from pathlib import Path
import pandas as pd
import os
from Misc import constant
from flask_cors import CORS, cross_origin
from flask import Response
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import shutil

#----------------------------Configurations for Flask------------------------------#

app = Flask(__name__)
app.secret_key = "secret key"
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = constant.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
web_url = "http://localhost:3000/"

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default=constant.DEFAULT_PATH + constant.INPUT_SUFFIX,
                help="path to (optional) input images file")
ap.add_argument("-o", "--output", type=str, default=constant.DEFAULT_PATH + constant.OUTPUT_SUFFIX,
                help="path to (optional) output images file")
ap.add_argument("-a", "--annotated", type=str, default=constant.DEFAULT_PATH + constant.ANNOTATED_OUTPUT_SUFFIX,
                help="path to (optional) annotated output images file")
ap.add_argument("-rt", "--retrain", type=str, default="False",
                help="whether or not model should be retrained")
ap.add_argument("-op", "--operation", type=str, default="Server",
                help="Local or Server")
ap.add_argument("-is", "--isolated", type=str, default="None",
                help="Isolated operation of module")
ap.add_argument("-eval", "--evaluated", type=str,
                default="None", help="Path to evalaluation csv file")
ap.add_argument("-eval-img", "--evaluated_image_path", type=str,
                default="None", help="Path to evalaluation image")
args = vars(ap.parse_args())
#----------------------------Assign req. arguments------------------------------#
output_path = args["output"]
input_path = args["input"]
annotated_output_path = args["annotated"]
operation = args["operation"]
isolated = args["isolated"]
evaluation_path = args["evaluated"]
evaluation_img_path = args["evaluated_image_path"]
#----------------------------Assign output path------------------------------#
pd_output_path = output_path + constant.PAPERDETCTION_FOLDER_SUFFIX
pd_annotated_output_path = annotated_output_path
pp_output_path = output_path + constant.PREPROCESSING_FOLDER_SUFFIX
td_output_path = output_path + constant.TEXTDETECTION_FOLDER_SUFFIX
td_annotated_output_path = annotated_output_path
adaptive_output_path = output_path + constant.ADAPTIVE_FOLDER_SUFFIX
tr_output_path = output_path + constant.TEXTRECOGNITION_FOLDER_SUFFIX
eval_output_path = output_path + constant.EVAL_FOLDER_SUFFIX
static_path = constant.DEFAULT_PATH + constant.STATIC_SUFFIX
#----------------------------Flask endpoints------------------------------#


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@cross_origin()
def hello_world():
    print("hello world")
    return {"hello": "world"}


@app.route('/input', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(web_url + "input")
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(web_url + "input")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(constant.UPLOAD_FOLDER, file.filename))
        print('upload_image filename: ' + filename)
        img_name = file.filename[:-4]
        flash('Image successfully uploaded and displayed below')
        return redirect(web_url + "input/" + img_name)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(web_url + "input")


@app.route("/manual_adaptive", methods=['POST'])
@cross_origin()
def run_adaptive_preprocesscing_manual():
    request_data = request.get_json()

    file_name = request_data['file_name']

    folder_name = file_name.split('_td_')[0]
    split = file_name.split('-denoised')
    split_name = split[0]
    split_suffix = split[1]

    path = os.path.join(td_output_path, folder_name, split_name + split_suffix)
    apply_CLAHE = request_data["apply_CLAHE"]
    window_size = request_data["window_size"]
    denoise_size = request_data["denoise_size"]
    clip_limit = request_data["clip_limit"]

    preview_file_name, blur = applyAdaptivePreprocesscingManualStep(
        path, adaptive_output_path, apply_CLAHE=apply_CLAHE, window_size=window_size, denoise_size=denoise_size, clip_limit=clip_limit)

    return {
        "file_name": preview_file_name,
    }


@app.route("/automatic_adaptive", methods=['POST'])
@cross_origin()
def run_adaptive_preprocesscing_automatic():
    request_data = request.get_json()
    file_names = request_data['file_names']

    for file_name in file_names:
        folder_name = file_name.split('_td_')[0]
        split = file_name.split('-denoised')
        split_name = split[0]
        split_suffix = ".jpg"

        path = os.path.join(td_output_path, folder_name,
                            split_name + split_suffix)
        after_adaptive_file_name, blur = applyAdaptivePreprocesscingStep(
            path, adaptive_output_path)

    return "Finished Apply Auto Adaptive Preprocessing"


@app.route("/get_blur/<file_name>")
@cross_origin()
def get_blur(file_name):
    # Opening JSON file
    json_file = open(adaptive_output_path + "/" + file_name + "/blur.json")

    #     # returns JSON object as
    #     # a dictionary
    json_file_data = json.load(json_file)
    print(json_file_data)
    return json_file_data


@ app.route('/directory_exist')
@ cross_origin()
def check_directory_exist():
    path = request.args.get('path')
    print(path)
    return {path: os.path.exists(path)}


@ app.route('/input_to_adaptive/<filename>/<isRerun>')
@ cross_origin()
def run_pipeline_to_adaptive(filename, isRerun):
    pd_path = pd_output_path + "/" + filename + "pd.jpg"
    pp_path = pp_output_path + "/" + filename + "pd_pp.jpg"
    td_dir_path = td_output_path + "/" + filename
    adaptive_dir_path = adaptive_output_path + \
        "/" + filename

    is_paper_detection_exist = os.path.exists(pd_path)
    is_preprocessing_exist = os.path.exists(pp_path)
    is_text_detection_folder_exist = os.path.isdir(td_dir_path)
    is_adaptive_folder_exist = os.path.isdir(adaptive_dir_path)

    if (isRerun == "false"):
        if (is_paper_detection_exist and is_preprocessing_exist and is_text_detection_folder_exist and is_adaptive_folder_exist):
            return "Completed running Paper Detection - Processcing - Text Detection - Adaptive for selected image"

    if (is_paper_detection_exist):
        os.unlink(pd_path)
        print("Is PD exist anymore ", os.path.exists(pd_path))
    if (is_preprocessing_exist):
        os.unlink(pp_path)
        print("Is PP exist anymore", os.path.exists(pp_path))
    if (is_text_detection_folder_exist):
        shutil.rmtree(td_dir_path)
        print("Is TD exist anymore", os.path.isdir(td_dir_path))
    if (is_adaptive_folder_exist):
        shutil.rmtree(adaptive_dir_path)
        print("Is Adaptive exist anymore", os.path.isdir(adaptive_dir_path))

    maskRCNN, fastRCNN = initDetectron2Module()

    # # =============================================================================
    # # Paper Detection and Preprocesscing
    # # =============================================================================

    img = input_path + "/" + filename + ".jpg"
    name = filename
    im = cv2.imread(img)

    # Encode the image as Base64
    # Encode the image as Base64
    with open(img, "rb") as img_file:
        data = base64.b64encode(img_file.read())

        # Paper Detection
        cropped_img, image_name = maskRCNN.predict(
            im=im, name=name, data=data)

        # Preprocesscing
        processed_img, processed_img_path = Amp.applyPreprocesscingStep(
            image_name=image_name, output_dir=pp_output_path)
        print('─' * 100)
        print("Preprocessing Image: " + processed_img_path)
    # ======================================================================================
    # Text Detection
    # ======================================================================================

    text_detection_folder = fastRCNN.predict(original=cropped_img,
                                             name=processed_img_path, data=data)
    print("Text Detection Finished - Result exported in : ",
          text_detection_folder)

    # ======================================================================================
    # Adaptive
    # ======================================================================================
    AdaptiveProcessing(text_detection_folder,
                       adaptive_output_path)
    return "Completed running Paper Detection - Processcing - Text Detection - Adaptive for selected image"


@app.route('/run_text_recognition/<filename>/<isRerun>')
@cross_origin()
def run_text_recognition(filename, isRerun="false"):

    vgg19_transformer = initTransformerModule()
    is_predict_exist, predict_info, is_eval_exist, eval_info = vgg19_transformer.predict(filename,
                                                                                         td_output_path, adaptive_output_path, tr_output_path, eval_output_path, is_rerun=isRerun)
    return {"predict_exist": is_predict_exist,
            "predict_info": predict_info,
            "eval_exist": is_eval_exist,
            "eval_info": eval_info}


@app.route("/text_recognition_eval", methods=['POST'])
@cross_origin()
def text_recognition_evaluation():
    cer = 1
    wer = 1

    request_data = request.get_json()

    ground_truths = request_data["ground_truths"]
    ground_truths = [s.lower() for s in ground_truths]

    predicts = request_data["predicts"]
    predicts = [s.lower() for s in predicts]

    cer = compute_accuracy(ground_truths, predicts, "cer")
    wer = compute_accuracy(ground_truths, predicts, "per_word")

    return {
        "wer": wer,
        "cer": cer
    }


@app.route('/')
@app.route('/get-input-list')
@cross_origin()
def get_input_list():
    combined_path = static_path + "input"
    img_list = get_img_list_from_directoty(combined_path)
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string


@app.route('/get-static-folder/<directory>')
@cross_origin()
def display_output_folder(directory):

    combined_path = static_path + "output" + "/" + directory
    subfolders = [str(os.path.basename(os.path.normpath(f)))
                  for f in os.scandir(combined_path) if f.is_dir()]
    print(subfolders)
    json_string = json.dumps(subfolders)
    return json_string


@app.route('/get-static-list/<directory>')
@cross_origin()
def display_output_list(directory):
    combined_path = static_path + "output" + "/" + directory
    img_list = get_img_list_from_directoty(combined_path)
    filenames = [str(Path(x).stem) for x in img_list]
    print(filenames)
    json_string = json.dumps(filenames)
    return json_string


@app.route('/get-static-list/<directory>/<category>')
@cross_origin()
def display_output_list_by_category(directory, category):
    combined_path = static_path + "output" + "/" + directory + "/" + category
    print(combined_path)
    img_list = get_img_list_from_directoty(combined_path)
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string


@app.route('/display-input/<name>')
@cross_origin()
def display_input_image(name):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='input/' + name), code=301)


@app.route('/display-adpt-denoised-output/<directory>/<category>/<filename>')
@cross_origin()
def display_adaptive_image(directory, category, filename):
    return redirect(url_for('static', filename='output/' + directory + '/denoised-output/' + category + '/' + filename), code=301)


@app.route('/display-output/<directory>/<filename>')
@cross_origin()
def display_ouptput_image(directory, filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='output/' + directory + '/' + filename), code=301)


@app.route('/get-static-denoised-list/<directory>/<category>')
@cross_origin()
def display_output_list_denoised(directory, category):
    combined_path = static_path + "output" + "/" + \
        directory + "/denoised-output/" + category
    print(combined_path)
    img_list = get_img_list_from_directoty(combined_path)
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string


@app.route('/display-sub-output/<directory>/<category>/<filename>')
@cross_origin()
def display_uncategorized_image(directory, category, filename):
    return redirect(url_for('static', filename='output/' + directory + '/' + category + '/' + filename), code=301)


def getImageUrl(path, name, category):
    if (category is None):
        url = "http://localhost:5000/display-output/${path}/${name}.jpg"
    else:
        url = "http://localhost:5000/display-sub-output/${path}/${category}/${name}.jpg"
    return url


def initModules():

    maskRCNN, fastRCNN = initDetectron2Module()

    vgg19_transformer = initTransformerModule()
    return maskRCNN, fastRCNN, vgg19_transformer


def initDetectron2Module():
    maskRCNN = PaperDetection.MaskCRNN(
        output_path=pd_output_path, annotated_output_path=pd_annotated_output_path)
    if(maskRCNN):
        print(" ✔ Paper Detection  -   MaskRCNN model loaded")
    else:
        raise ValueError(
            '❌ Paper Detection - MaskRCNN model failed to load')

    fastRCNN = TextDetection_Detectron2.FasterRCNN(
        output_path=td_output_path, annotated_output_path=td_output_path)
    if(fastRCNN):
        print(" ✔ Text Detection   -   FastRCNN model loaded")
    else:
        raise ValueError(
            '❌ Text Detection - FastRCNN model failed to load')
    return maskRCNN, fastRCNN


def initTransformerModule():
    vgg19_transformer = TextRecognition()
    if(vgg19_transformer):
        print(" ✔ Text Recognition   -   VGG19-Transormer model loaded")
    else:
        raise ValueError(
            '❌ Text Recognition - VGG19-Transormer model failed to load')
    return vgg19_transformer


def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)


if __name__ == '__main__':

    print("#----------------------------DOCR - OUCRU Handwriting Recognition - Main ------------------------------#")
    print("# " + operation)
    if operation == "Local":

        maskRCNN, fastRCNN, vgg19_transformer = initModules()

        records_count = 0
        img_list = get_img_list_from_directoty(input_path)
        filenames = [str(Path(x).stem) for x in img_list]
        # =============================================================================
        # Paper Detection and Preprocesscing
        # =============================================================================

        for img, name in zip(img_list, filenames):
            im = cv2.imread(img)

            # Encode the image as Base64
            with open(img, "rb") as img_file:
                data = base64.b64encode(img_file.read())

                # Paper Detection
                cropped_img, image_name = maskRCNN.predict(
                    im=im, name=name, data=data)

                # Preprocesscing
                processed_img, processed_img_path = Amp.applyPreprocesscingStep(
                    image_name=image_name, output_dir=pp_output_path)
                print('─' * 100)
                print("Preprocessing Image: " + processed_img_path)
        # ======================================================================================
        # Text Detection
        # ======================================================================================

            text_detection_folder = fastRCNN.predict(original=cropped_img,
                                                     name=processed_img_path, data=data)
            print("Text Detection Finished - Result exported in : ",
                  text_detection_folder)

        # ======================================================================================
        # Adaptive
        # ======================================================================================
            AdaptiveProcessing(text_detection_folder,
                               adaptive_output_path)

        # ======================================================================================
        # Text Recognition
        # ======================================================================================
        tr_img_directories = get_img_list_from_directoty(
            adaptive_output_path)
        for directory in tr_img_directories:
            filename = os.path.split(directory)[1]
            is_predict_exist, predict_info, is_eval_exist, eval_info = vgg19_transformer.predict(filename,
                                                                                                 td_output_path, adaptive_output_path, tr_output_path, eval_output_path)

            if (is_eval_exist):
                predicts = map(lambda x: x['predict'], predict_info)
                ground_truth = map(lambda x: x['ground_truth'], eval_info)
                wer, cer = evaluation(ground_truth, predicts)
                print_msg_box(msg="WER: " + str(wer) + " - CER: " + str(cer),
                              indent=5, title="Evaluation Error Rate")
        # else:
        #     print("Null found at: ", image_name)
    elif operation == "Server":
        app.run(host="0.0.0.0", port=5000, debug=False,
                threaded=True, use_reloader=False)
        # print("# Pipeline finished " + operation +
        #       " with " + str(records_count) + " medical records")
