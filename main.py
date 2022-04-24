import argparse
from PaperDetection import PaperDetectionAndCorrection
import sys
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
    PaperDetectionAndCorrection.paperDetectionOperation(operation,input_path,output_path,annotated_output_path)