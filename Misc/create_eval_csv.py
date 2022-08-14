
import argparse
import pandas as pd
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-eval", "--evaluated", type=str,
                default="None", help="Path to original evalaluation csv file to be")
args = vars(ap.parse_args())
evaluation_csv_path = args["evaluated"]

annotaiton_file_path = "mnt/d/DOCR/OUCRU-Handwriting-Pipeline/TextRecognition/vietocr/Sorted/train_resorted_adaptive.txt"
# Print it out if you want
if __name__ == '__main__':
    with open(evaluation_csv_path, 'r') as eval_file:
        datareader = csv.reader(eval_file)
        for index, image_name, min_x, min_y, max_x, max_y, original_image_name in datareader:
            print(index, image_name, min_x, min_y,
                  max_x, max_y, original_image_name)
