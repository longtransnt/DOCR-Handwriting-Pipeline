
import csv
import json
import os
import pandas as pd

# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     
   df = pd.read_csv(csvFilePath)
   df = df[["image_name", "min_x","min_y","max_x","max_y","original_image_name"]]
#    print(df)
#    df.drop("index, axis=1, inplace=True)
#    df.drop("Unnamed: 0", axis=1, inplace=True)
   df.reset_index().to_json(jsonFilePath,orient='records')
         
# Driver Code
 
# Decide the two file paths according to your
# computer system
csvFilePath = r'/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output/TextDetection/Sorted/coordinate/coordinate_csv/'
jsonFilePath = r'/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output/TextDetection/Sorted/coordinate/coordinate_json/'
 
# Call the make_json function
srcfiles = os.listdir(csvFilePath)
for filename in srcfiles :
    make_json(csvFilePath + filename, jsonFilePath + filename[:-4] + ".json")