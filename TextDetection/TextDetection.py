from pathlib import Path
from mmocr.utils.ocr import MMOCR
from td_model import textdet_models
from mmocr.datasets.pipelines.crop import crop_img
from pathlib import Path
import pandas as pd


def predict(input_path, output_path):
    for model in textdet_models.keys():
        print("model:", model)
        
        Path(output_path + model).mkdir(parents=True, exist_ok=True)
        try:
            mmocr = MMOCR(det=model, recog=None, device='cpu')
            _ = mmocr.readtext(input_path, output = output_path + model)
        except Exception as e:
            print(e)

def export_coordinates_to_csv(input_path, output_path):
    end2end_res = []
    # # Find bounding boxes in the images (text detection)
    # det_result = self.single_inference(det_model, self.args.arrays,
    #                                    self.args.batch_mode,
    #                                    self.args.det_batch_size)
    bboxes_list = [res['boundary_result'] for res in det_result]


    input_image_path = input_path + image_name_for_test
    img_list = [input_image_path]

    arrays = [mmcv.imread(x) for x in img_list]
    filenames = [str(Path(x).name) for x in img_list]
    cropped_images = []

    print('img_list:', *img_list, sep='\n\t')
    print('arrays size:', len(arrays))
    print('filenames:', filenames)
    print("--------------------")

    # Create a dataframe to export coordiantes of boxes to csv file
    box_column_names = ['image_name', 'min_x', 'min_y', 'max_x', 'max_y']
    boxes_coordinates = pd.DataFrame(columns=box_column_names)

    for filename, arr, bboxes in zip(filenames, arrays, bboxes_list):
        box_imgs = []
        print("filename: ", filename)
        print('bboxes size:', len(bboxes))
    
    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:-1]]
        box_res['box_score'] = float(bbox[-1])
        # box = bbox[:8]
        if len(bbox) > 9:
        min_x = min(bbox[0:-1:2])
        min_y = min(bbox[1:-1:2])
        max_x = max(bbox[0:-1:2])
        max_y = max(bbox[1:-1:2])
        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        
        # Append coordinates of a box to data frame
        box_coordinates = pd.Series([filename, min_x, min_y, max_x, max_y], index=box_column_names)
        boxes_coordinates = boxes_coordinates.append(box_coordinates, ignore_index=True)

        box_img = crop_img(arr, box)
        box_imgs.append(box_img)

        cropped_images.append(box_imgs)


    print("--------------------")
    boxes_coordinates = boxes_coordinates.astype(int, errors='ignore')
    print('boxes_coordinates:\n', boxes_coordinates)

    # Export coordinates of boxes to csv
    boxes_coordinates.to_csv(output_path + 'boxes_coordinates.csv')
