
import csv
import json
 
 
# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     
    csvfile = open(csvFilePath, 'r')
    jsonfile = open(jsonFilePath, 'w')

    fieldnames = ('image_name', 'min_x', 'min_y', 'max_x', 'max_y', "original_image_name")
    reader = csv.DictReader( csvfile, fieldnames)
    out = json.dumps( [ row for row in reader ] )
    jsonfile.write(out)
         
# Driver Code
 
# Decide the two file paths according to your
# computer system
csvFilePath = r'/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output/TextDetection/boxes_coordinates__.csv'
jsonFilePath = r'/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output/TextDetection/boxes_coordinates__.json'
 
# Call the make_json function
make_json(csvFilePath, jsonFilePath)