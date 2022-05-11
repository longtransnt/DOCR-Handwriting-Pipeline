import argparse
from PaperDetection import PaperDetectionAndCorrection
from Preprocessing import AmplifyClaheDeblureSauvolaDenoiseConnCompo as Amp
import sys
import base64
import cv2
import numpy as np
from pathlib import Path
import os
#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Input",
	help="path to (optional) input images file")
ap.add_argument("-o", "--output", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output",
	help="path to (optional) output images file")
ap.add_argument("-a", "--annotated", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Annotated_Output",
help="path to (optional) annotated output images file")
ap.add_argument("-rt", "--retrain", type=str, default="False",
	help="whether or not model should be retrained")
ap.add_argument("-op", "--operation", type=str, default="Predict",
help="Predict or Annotation")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#
output_path = args["output"]
input_path = args["input"]
annotated_output_path = args["annotated"]
operation = args["operation"]


if __name__ == '__main__':
    
	imgs_dir = [
        #  r"/home/longtrans/OUCRU-Handwriting-Recognition-reference/Paper_Detection/bill_demo/Subset2",
        r"{}".format(input_path),
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

	filenames = [str(Path(x).stem) for x in img_list]
	pd_output_path = output_path + '/PaperDetection'
	mask = PaperDetectionAndCorrection.MaskCRNN(output_path=pd_output_path, annotated_output_path = annotated_output_path)
    
	pp_output_path = output_path + '/Pre'
	for img, name in zip(img_list, filenames):
        # im = cv2.imread()
		im = cv2.imread(img)

        # # Encode the image as Base64
		with open(img, "rb") as img_file:
			data = base64.b64encode(img_file.read())

		
		if operation == "Predict":
			image_name = mask.predict(im=im, name=name, data=data)
			if(image_name is not None):
				Amp.applyPreprocesscingStep(image_name = image_name, output_dir=pp_output_path)
			else: 
				print(image_name)
		elif operation == "Annotation":
			cropped = mask.annotate(im=im, name=name, data=data)
		else: 
			exit

		
