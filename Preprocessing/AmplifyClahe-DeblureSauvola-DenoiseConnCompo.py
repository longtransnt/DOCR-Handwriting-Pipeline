# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:45:22 2022

@author: ASUS
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola


input_dir = './med_records/'
output_dir = './med_records_output_clahe_sauvola_dashes/'

# image_name = '21.000051 (6).jpg'
image_name = '21.000179 (34).jpg'
# image_name = '21.000179 (53).jpg'
# image_name = 'out_21,.001477 (11).jpg'


output_dir += image_name[:-4] + '/'

# =============================================================================
# Create output directory
# =============================================================================
Path(output_dir).mkdir(parents=True, exist_ok=True)

img = cv2.imread(input_dir + image_name, cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =============================================================================
# create a CLAHE object (Arguments are optional).
# =============================================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)
# cv2.imwrite(output_dir + image_name[:-4] + '--amplified-CLAHE.png', clahe_image)


# =============================================================================
# Sauvola Thresholding
# =============================================================================
window_size = 25
thresh_sauvola = threshold_sauvola(clahe_image, window_size=window_size)
binary_sauvola = clahe_image > thresh_sauvola
sauvola_image = np.uint8(binary_sauvola * 255)

# =============================================================================
# INVERSE thresholding
# =============================================================================
th, sauvola_bin_image = cv2.threshold(sauvola_image, 0, 255, cv2.THRESH_BINARY_INV)

# =============================================================================
# Denoise
# =============================================================================
def remove_small_objects(img, min_size):
        # find all your connected components (white blobs in your image)
        connectivity = 8
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                                img, connectivity, cv2.CV_32S)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1] 
        nb_components = nb_components - 1
        
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 0
        
        return 255-img2
        # res = cv2.bitwise_not(img2)
        # return res


for size in range(15, 30, 1):
    img_denoised = remove_small_objects(sauvola_bin_image, size)
    img_denoised = img_denoised
    cv2.imwrite(output_dir + image_name[:-4] + '--Denoised-size-'+str(size)+'.png', img_denoised)


