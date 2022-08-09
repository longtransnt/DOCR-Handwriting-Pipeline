import argparse
from unittest import skip
from PaperDetection import PaperDetection
from Preprocessing import AmplifyClaheDeblureSauvolaDenoiseConnCompo as Amp
from TextDetection import TextDetection_Detectron2
from TextDetection_PostProcesscing.AdaptivePreprocesscing import applyAdaptivePreprocesscingStep
from TextRecognition.vietocr.TextRecognition import TextRecognition
import sys
import base64
import cv2
import numpy as np
import datetime
from pathlib import Path
import pandas as pd
import os
from Misc import constant
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
ap.add_argument("-op", "--operation", type=str, default="Predict",
                help="Predict or Annotation")
ap.add_argument("-is", "--isolated", type=str, default="None",
                help="Isolated operation of module")
args = vars(ap.parse_args())
#----------------------------Assign req. arguments------------------------------#
output_path = args["output"]
input_path = args["input"]
annotated_output_path = args["annotated"]
operation = args["operation"]
isolated = args["isolated"]
#----------------------------Assign output path------------------------------#
pd_output_path = output_path + constant.PAPERDETCTION_FOLDER_SUFFIX
pd_annotated_output_path = annotated_output_path
pp_output_path = output_path + constant.PREPROCESSING_FOLDER_SUFFIX
td_output_path = output_path + constant.TEXTDETECTION_FOLDER_SUFFIX
td_annotated_output_path = annotated_output_path
adaptive_output_path = output_path + constant.ADAPTIVE_FOLDER_SUFFIX
tr_output_path = output_path + constant.TEXTRECOGNITION_FOLDER_SUFFIX


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
                '❌ Text Detection - VGG19-Transormer model failed to load')

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
                if(image_name is not None):

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

                    cropped_img_list = get_img_list_from_directoty(
                        text_detection_folder)
                    cropped_filenames = [str(Path(x).stem)
                                         for x in cropped_img_list]

                    for cropped_img, cropped_filename in zip(cropped_img_list, cropped_filenames):
                        if(cropped_img.endswith(".csv")):
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
        tr_img_list = get_img_list_from_directoty(
            adaptive_output_path + '/denoised/')
        tr_filenames = [str(Path(x).stem) for x in tr_img_list]
        dashed_line = '=' * 100
        head = f'{"filename":35s}\t' \
            f'{"predicted_string (non-bigram)":35s}\t' \
            f'{"predicted_string (bigram)":35s}'

        text_recognition_csv_headers = [
            'filename', 'predicted_string (non-bigram)', 'predicted_string (bigram)']
        text_recognition_csv = pd.DataFrame(
            columns=text_recognition_csv_headers)

        print(f'{dashed_line}\n{head}\n{dashed_line}')

        for img, name in zip(tr_img_list, tr_filenames):
            prediction = vgg19_transformer.infer(img)
            row_output = f'{name:20s}\t{prediction:35s}' \
                f'\t{prediction:35s}'
            print(row_output)
            row = pd.Series([name, prediction, prediction],
                            index=text_recognition_csv_headers)
            text_recognition_csv = pd.concat([text_recognition_csv,
                                              row], axis=0)
        date_time = datetime.datetime.now()
        text_recognition_csv.to_csv(
            tr_output_path + "/" + str(date_time) + '.csv')
    else:
        exit

    print("# Pipeline finished " + operation +
          " with " + str(records_count) + " medical records")
