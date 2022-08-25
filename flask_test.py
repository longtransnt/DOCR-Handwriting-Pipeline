import pandas as pd
from pathlib import Path
import os
from Misc import constant
from flask_cors import CORS, cross_origin
from flask import Response
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from TextDetection_PostProcesscing.AdaptivePreprocesscing import applyAdaptivePreprocesscingStep, applyAdaptivePreprocesscingManualStep, denoise
from werkzeug.utils import secure_filename
import json
import numpy as np
from Misc import constant
from TextRecognition.vietocr.vietocr.tool.utils import compute_accuracy

static_path = "/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/static/"
output_path = static_path + "output"
input_path = static_path + "input"

UPLOAD_FOLDER = static_path + "uploads"
app = Flask(__name__)
app.secret_key = "secret key"
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
td_output_path = output_path + constant.TEXTDETECTION_FOLDER_SUFFIX
adaptive_output_path = output_path + constant.ADAPTIVE_FOLDER_SUFFIX


def get_img_list_from_directoty(input):
    imgs_dir = [
        r"{}".format(input),
    ]

    img_list = []
# Loading Image List
    for img in imgs_dir:
        print(img)
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


@app.route('/display-output/<directory>/<filename>')
@cross_origin()
def display_ouptput_image(directory, filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='output/' + directory + '/' + filename), code=301)


@app.route('/display-sub-output/<directory>/<category>/<filename>')
@cross_origin()
def display_uncategorized_image(directory, category, filename):
    return redirect(url_for('static', filename='output/' + directory + '/' + category + '/' + filename), code=301)


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

    preview_file_name = applyAdaptivePreprocesscingManualStep(
        path, adaptive_output_path, apply_CLAHE=apply_CLAHE, window_size=window_size, denoise_size=denoise_size, clip_limit=clip_limit)

    return {
        "file_name": preview_file_name,
    }


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


def getImageUrl(path, name, category):
    if (category is None):
        url = "http://localhost:5000/display-output/${path}/${name}.jpg"
    else:
        url = "http://localhost:5000/display-sub-output/${path}/${category}/${name}.jpg"
    return url


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True,
            threaded=True, use_reloader=False)
