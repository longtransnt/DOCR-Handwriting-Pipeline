Labelme.exe - Manual Annotation Tool for marking annotation on image data
-> Result: JSON file with Annotation of that image

Using Coco Label_Me to export the data into a form of COCO JSON file that can be used to train Detectron2
1. Put the folder into your Ubuntu environment
2. cd root (folder)
3. Run python3 datasets/coco_labelme.py Subset

Subset: set of images that being used to build the coco json file, need to be annotated before
Datasets/coco_labelme.py: The file code that being used to generate coco json file
Output: trainval.json
Library needed: Labelme

pip install labelme

GitHub and more info:
https://github.com/Paperspace/object-detection-segmentation