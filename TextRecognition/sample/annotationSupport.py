from sys import path_hooks
import cv2
import os
import re
import codecs

# change "test" to the specific folder name: for eg "2"
rootdir = '/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/TextRecognition/sample/7'

# change name to match with folder
f = open('7_annotated.txt', 'a')

for subdir, dirs, files in os.walk(rootdir):

    for file in files:
        path = os.path.join(subdir, file)
        subpath = os.path.join(*(path.split(os.path.sep)[7:]))
        print(subpath)
        # im = cv2.imread(path)
        # cv2.imshow("paper", im)
        truth = input("What is the text: ")
        
        f.write(subpath + " " + truth + "\n")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
f.close()
        



