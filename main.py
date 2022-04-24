import argparse

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Input",
	help="path to (optional) input images file")
ap.add_argument("-o", "--output", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output",
	help="path to (optional) output images file")
ap.add_argument("-oa", "--output_annotated", type=str, default="/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Annotated_Output",
help="path to (optional) annotated output images file")
ap.add_argument("-rt", "--retrain", type=str, default="False",
	help="whether or not model should be retrained")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#