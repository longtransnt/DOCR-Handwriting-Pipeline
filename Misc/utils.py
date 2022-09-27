from gettext import npgettext
from pathlib import Path


def get_img_list_from_directoty(input):
    imgs_dir = [
        r"{}".format(input),
    ]

    img_list = []
# Loading Image List
    for img in imgs_dir:
        # Create a list of the images
        if isinstance(img, str):
            img_path = Path(img)
            if img_path.is_dir():
                img_list += [str(x) for x in img_path.glob('*')]
            else:
                img_list += [str(img_path)]
        elif isinstance(img, npgettext.ndarray):
            img_list += [img]
    return img_list
