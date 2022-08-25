import argparse
from operator import contains
from unittest import skip
from PaperDetection import PaperDetection
from Preprocessing import AmplifyClaheDeblureSauvolaDenoiseConnCompo as Amp
from TextDetection import TextDetection_Detectron2
from TextDetection_PostProcesscing.AdaptivePreprocesscing import applyAdaptivePreprocesscingStep, applyAdaptivePreprocesscingManualStep, denoise
from TextRecognition.vietocr.TextRecognition import TextRecognition
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


#----------------------------Configurations for Flask------------------------------#
UPLOAD_FOLDER = '/mnt/d/OUCRU-Handwriting-Recognition/static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

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
                help="Predict or Annotation")
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


#----------------------------Flask endpoints------------------------------#
@app.route('/')
@cross_origin()
def hello_world():
    print("hello world")
    return {"hello": "world"}


@app.route('/input', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


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

    applyAdaptivePreprocesscingManualStep(
        path, adaptive_output_path, apply_CLAHE=apply_CLAHE, window_size=window_size, denoise_size=denoise_size)

    return {
        "file_name": path,
        "apply_CLAHE": apply_CLAHE,
        "window_size": window_size,
        "denoise_size": denoise_size
    }


@ app.route('/display/<filename>')
@ cross_origin()
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='Output/TextDetection/21.000440 (33)pdpd/' + filename), code=301)


@ app.route('/directory_exist')
@ cross_origin()
def check_directory_exist():
    path = request.args.get('path')
    print(path)
    return {path: os.path.exists(path)}


def get_img_list_from_directoty(input):
    imgs_dir = [
        r"{}".format(input),
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
    return img_list


if __name__ == '__main__':

    print("#----------------------------DOCR - OUCRU Handwriting Recognition - Main ------------------------------#")
    print("# " + operation)
    if operation == "Predict":

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

        vgg19_transformer = TextRecognition()
        if(vgg19_transformer):
            print(" ✔ Text Recognition   -   VGG19-Transormer model loaded")
        else:
            raise ValueError(
                '❌ Text Recognition - VGG19-Transormer model failed to load')

        records_count = 0
        img_list = get_img_list_from_directoty(input_path)
        filenames = [str(Path(x).stem) for x in img_list]
        # # =============================================================================
        # # Paper Detection and Preprocesscing
        # # =============================================================================
        for img, name in zip(img_list, filenames):
            im = cv2.imread(img)

            # Encode the image as Base64
            with open(img, "rb") as img_file:
                data = base64.b64encode(img_file.read())

                # Paper Detection
                cropped_img, image_name = maskRCNN.predict(
                    im=im, name=name, data=data)
                if(image_name is not None):

                    # Preprocesscing
                    processed_img, processed_img_path = Amp.applyPreprocesscingStep(
                        image_name=image_name, output_dir=pp_output_path)
                    print('─' * 100)
                    print("Preprocessing Image: " + processed_img_path)
        # #     # ======================================================================================
        # #     # Text Detection
        # #     # ======================================================================================
            text_detection_folder = fastRCNN.predict(original=cropped_img,
                                                     name=processed_img_path, data=data)
            print("Text Detection Finished - Result exported in : ",
                  text_detection_folder)

            cropped_img_list = get_img_list_from_directoty(
                text_detection_folder)
            cropped_filenames = [str(Path(x).stem)
                                 for x in cropped_img_list]

            for cropped_img, cropped_filename in zip(cropped_img_list, cropped_filenames):
                if(cropped_img.endswith(".csv") or cropped_img.endswith(".json") or "visualize" in cropped_img):
                    continue
                applyAdaptivePreprocesscingStep(
                    cropped_img, adaptive_output_path)
            print('─' * 100)
            # td.operation(input_path = td_input)
            records_count += 1
        else:
            print("Null found at: ", image_name)
        img_list = get_img_list_from_directoty(input_path)

        # ======================================================================================
        # Text Recognition
        # ======================================================================================

        tr_img_directories = get_img_list_from_directoty(
            adaptive_output_path + '/denoised-output/')
        for directory in tr_img_directories:
            text_recognition_json_result = []

            path, dir_name = os.path.split(directory)

            # Opening JSON file
            json_file = open(td_output_path + "/" + dir_name +
                             "pdpd/" + "coordinates.json")

            # returns JSON object as
            # a dictionary
            json_file_data = json.load(json_file)

            tr_img_list = get_img_list_from_directoty(directory)
            tr_filenames = [str(Path(x).stem) for x in tr_img_list]

            dashed_line = '=' * 100
            head = f'{"filename":35s}\t' \
                f'{"predicted_string (non-bigram)":35s}\t' \
                f'{"predicted_string (bigram)":35s}'

            text_recognition_csv_headers = [
                'filename', 'predicted_string (non-correction)', 'predicted_string (correction)']
            text_recognition_csv = pd.DataFrame(
                columns=text_recognition_csv_headers)

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            for img, name in zip(tr_img_list, tr_filenames):
                split = name.split('-denoised')
                split_name = split[0]
                split_suffix = split[1]
                cor_dict = list(
                    filter(lambda line: line['image_name'].split('.jpg')[0] == split_name, json_file_data))

                prediction, correction = vgg19_transformer.infer(img)
                row_output = f'{name:20s}\t{prediction:35s}' \
                    f'\t{correction:35s}'
                print(row_output)
                en = correction.encode("utf8")
                print(en)
                print(en.decode("utf8"))

                cor_dict[0]["ground_truth"] = en.decode("utf8")

                text_recognition_json_result.append(cor_dict[0])

            base = Path(tr_output_path)
            jsonpath = base / (name + ".json")
            jsonpath.write_text(json.dumps(text_recognition_json_result))

    elif operation == "Eval":
        if evaluation_path == "None":
            raise ValueError(
                'Need Evaluation csv path to be specified')
        if evaluation_img_path == "None":
            raise ValueError(
                'Need Evaluation img path to be specified')
    elif operation == "Server":
        app.run(host="0.0.0.0", port=5000, debug=True,
                threaded=False, use_reloader=False)
        # print("# Pipeline finished " + operation +
        #       " with " + str(records_count) + " medical records")


def PaperDetectionCaller(img, name):
    maskRCNN = PaperDetection.MaskCRNN(
        output_path=pd_output_path, annotated_output_path=pd_annotated_output_path)

    if(maskRCNN):
        print(" ✔ Paper Detection  -   MaskRCNN model loaded")
    else:
        raise ValueError(
            ' ❌ Paper Detection    -   MaskRCNN model failed to load')

    cropped_img, image_name = maskRCNN.predict(im=img, name=name)
    return cropped_img


def PreprocessingCaller(img_path):
    processed_img, processed_img_path = Amp.applyPreprocesscingStep(
        image_name=img_path, output_dir=pp_output_path)
    return processed_img


def TextDetectionCaller(img, name):
    fastRCNN = TextDetection_Detectron2.FasterRCNN(
        output_path=td_output_path, annotated_output_path=td_output_path)

    if(fastRCNN):
        print(" ✔ Text Detection   -   FastRCNN model loaded")
    else:
        raise ValueError(
            '   ❌ Text Detection   -   FastRCNN model failed to load')

    # TODO: edit to Return records with bounding box
    text_detection_folder = fastRCNN.predict(original=cropped_img,
                                             name=processed_img_path, data=data)
    return text_detection_folder


def AdaptivePreprocesscingCaller():
    return "Not implemented yet"
