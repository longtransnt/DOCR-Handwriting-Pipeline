import io
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance


def parse_image(contents, filename, date):
    # Take uploaded image, from dcc upload, convert to np array, and reshape
    # for nn
    content_type, content_string = contents.split(",")
    im = Image.open(io.BytesIO(base64.b64decode(content_string)))
    open_cv_image = np.array(im)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    array[np.where(array == 0)] = 255

    im_pil = Image.fromarray(array)

    enhancer = ImageEnhance.Sharpness(im_pil)
    enhanced_im = enhancer.enhance(10.0)

    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return "data:image/png;base64," + im_b64


def create_img(arr, shape=(28, 28)):
    arr = arr.reshape(shape).astype(np.float64)
    image_b64 = numpy_to_b64(arr)
    return image_b64
