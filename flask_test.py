import pandas as pd
from pathlib import Path
import os
from Misc import constant
from flask_cors import CORS, cross_origin
from flask import Response
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json

output_path = "/mnt/d/OUCRU-Handwriting-Recognition/static/Output/"
input_path = "/mnt/d/OUCRU-Handwriting-Recognition/static/Input/"
static_path = "/mnt/d/OUCRU-Handwriting-Recognition/static/"

UPLOAD_FOLDER = '/mnt/d/OUCRU-Handwriting-Recognition/static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

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

@app.route('/')

@app.route('/get-input-list')
@cross_origin()
def get_input_list():
    combined_path= static_path + "input"
    img_list = get_img_list_from_directoty(combined_path) 
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string

@app.route('/get-static-list/<directory>')
@cross_origin()
def display_output_list(directory):
    combined_path= static_path + "output" + "/" + directory
    img_list = get_img_list_from_directoty(combined_path) 
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string

@app.route('/get-static-list/<directory>/<category>')
@cross_origin()
def display_output_list_by_category(directory, category):
    combined_path= static_path + "output" + "/" + directory + "/" + category
    img_list = get_img_list_from_directoty(combined_path) 
    filenames = [str(Path(x).stem) for x in img_list]
    json_string = json.dumps(filenames)
    return json_string

@app.route('/display-input')
@cross_origin()
def display_input_image(path, filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='input/' + filename), code=301)

@app.route('/display-output/<directory>/<filename>')
@cross_origin()
def display_ouptput_image(directory, filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='output/' + directory + '/' + filename), code=301)

@app.route('/display-sub-output/<directory>/<category>/<filename>')
@cross_origin()
def display_uncategorized_image(directory, category, filename):
    return redirect(url_for('static', filename='output/' + directory + '/' + category + '/' + filename), code=301)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False, use_reloader=False)
