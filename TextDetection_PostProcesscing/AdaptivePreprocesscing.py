import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import os_sorted
from pathlib import Path
from skimage.filters import threshold_sauvola

# =============================================================================
# Blur Detection
# =============================================================================


def detect_blur_fft(image, size=15, thresh=10, vis=False):
    # convert to gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    h, w = gray.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more easy to analyze
    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        # ax[0].imshow(image, cmap="gray")
        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        # ax[1].imshow(magnitude, cmap="gray")
        ax[1].imshow(magnitude)
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low frequencies),
    # apply the inverse shift such that the DC component once again becomes
    # the top-left, and then apply the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value

    return (mean, mean <= thresh)

# =============================================================================
# Denoise
# =============================================================================


def remove_small_objects(img, min_size):
    # find all your connected components (white blobs in your image)
    connectivity = 8
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity, cv2.CV_32S)
    # connectedComponentswithStats yields every seperated component with information on each of them,
    # such as size the following part is just taking out the background which is also considered a component,
    # but most of the time we don't want that.
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


def denoise(output_dir_denoised, img_path, blur_degree=33, tile_size=(8, 8), vis=False, scale=1):
    # =============================================================================
    # Read input image
    # =============================================================================
    if vis:
        print('ORIGINAL:')
        # cv2_imshow(cv2.resize(image, (0,0), fx=scale, fy=scale))

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('GRAY:')
    # cv2_imshow(gray)
    # =============================================================================
    # create a CLAHE object (Arguments are optional).
    # =============================================================================
    if blur_degree <= 33:  # default blur_degree <= 33
        # clipLimit range (5:11)
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=tile_size)
        clahe_image = clahe.apply(gray)
        # cv2.imwrite(output_dir + image_name[:-4] + '--amplified-CLAHE.png', clahe_image)
        if vis:
            print('CLAHE:')
            # cv2_imshow(clahe_image)
    else:
        if vis:
            print('CLAHE: NO')
        clahe_image = gray
    # =============================================================================
    # Sauvola Thresholding
    # =============================================================================
    window_size = 51  # odd number
    thresh_sauvola = threshold_sauvola(clahe_image, window_size=window_size)
    binary_sauvola = clahe_image > thresh_sauvola
    sauvola_image = np.uint8(binary_sauvola * 255)

    if vis:
        print('SAUVOLA: window size =', window_size)
        # cv2_imshow(sauvola_image)

    # Split image name
    image_name = Path(img_path).stem
    image_suffix = Path(img_path).suffix
    # cv2.imwrite(str(output_dir_sauvola / image_name)+'-SauvolaWS='+str(window_size) + image_suffix, sauvola_image)
    # =============================================================================
    # INVERSE thresholding
    # =============================================================================
    th, sauvola_bin_image = cv2.threshold(
        sauvola_image, 0, 255, cv2.THRESH_BINARY_INV)

    img_denoised = remove_small_objects(sauvola_bin_image, blur_degree/2)

    if vis:
        print('DENOISED:')
        # cv2_imshow(img_denoised)
        print('-'*80)

    original_name = image_name.split('pdpd')[0]

    print(original_name)
    isExist = os.path.exists(
        str(output_dir_denoised) + "/" + original_name)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(str(output_dir_denoised) + "/" + original_name)
        print("The new directory is created for " + original_name)

    cv2.imwrite(str(output_dir_denoised) + "/" + original_name + "/" +
                str(image_name) + '-denoised' + image_suffix, img_denoised)

    return img_denoised


def applyAdaptivePreprocesscingStep(image_path, output_dir):
    image_path = image_path.strip()
    # print(*images_name, sep='\n')
    SIZE = 15  # SIZE in range (10, 31); best range (10, 21)
    output_dir = Path(output_dir)
    output_dir_denoised = output_dir / 'denoised-output'

    Path.mkdir(output_dir, exist_ok=True)
    Path.mkdir(output_dir_denoised, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # (mean, blurry) = detect_blur_fft(image=image, size=SIZE, thresh=THRESHOLD, vis=False)
    (mean, blurry) = detect_blur_fft(image=image, size=SIZE)
    # indication = "BLURRY" if blurry else "Clear"
    # print('{:>25} --- {:8.3f} --- {}'.format(img_name, mean, indication))
    image_name = Path(image_path).stem
    print(
        '☒ Applied Adaptive PP: {:>25}\t---\tFFT Metric: {:8.3f}'.format(image_name, mean))

    # for tileGridSize in range():
    image_denoise = denoise(output_dir_denoised,
                            image_path, blur_degree=mean, vis=False)

    # print('DENOISED:')
    # scale = 1
    # cv2_imshow(cv2.resize(image_denoise, (0, 0), fx=scale, fy=scale))


def applyAdaptivePreprocesscingManualStep(image_path, output_dir, apply_CLAHE, window_size, denoise_size):
    image_path = image_path.strip()
    # print(*images_name, sep='\n')
    SIZE = 15  # SIZE in range (10, 31); best range (10, 21)
    output_dir = Path(output_dir)
    output_dir_denoised = output_dir / 'denoised-output'

    Path.mkdir(output_dir, exist_ok=True)
    Path.mkdir(output_dir_denoised, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # (mean, blurry) = detect_blur_fft(image=image, size=SIZE, thresh=THRESHOLD, vis=False)
    (mean, blurry) = detect_blur_fft(image=image, size=SIZE)
    # indication = "BLURRY" if blurry else "Clear"
    # print('{:>25} --- {:8.3f} --- {}'.format(img_name, mean, indication))
    image_name = Path(image_path).stem
    print('☒ Applied Adaptive PP: {:>25}\t---\tFFT Metric: {:8.3f} - Parameters = apply_CLAHE: {} window_size: {:8.3f} - denoised_size: {:8.3f}'.format(
        image_name, mean, apply_CLAHE, window_size, denoise_size))

    image_denoise = denoise_manual(output_dir_denoised,
                                   image_path, blur_degree=mean, apply_CLAHE=apply_CLAHE, window_size=window_size, denoised_size=denoise_size)

    # print('DENOISED:')
    # scale = 1
    # cv2_imshow(cv2.resize(image_denoise, (0, 0), fx=scale, fy=scale))


def denoise_manual(output_dir_denoised, img_path, blur_degree, apply_CLAHE=True, tile_size=(8, 8), scale=1, window_size=51, denoised_size=12):
    # =============================================================================
    # Read input image
    # =============================================================================

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('GRAY:')
    # cv2_imshow(gray)
    # =============================================================================
    # create a CLAHE object (Arguments are optional).
    # =============================================================================
    if apply_CLAHE:
        # clipLimit range (5:11)
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=tile_size)
        clahe_image = clahe.apply(gray)
        # cv2.imwrite(output_dir + image_name[:-4] + '--amplified-CLAHE.png', clahe_image)
    else:
        clahe_image = gray
    # =============================================================================
    # Sauvola Thresholding
    # =============================================================================
    thresh_sauvola = threshold_sauvola(clahe_image, window_size=window_size)
    binary_sauvola = clahe_image > thresh_sauvola
    sauvola_image = np.uint8(binary_sauvola * 255)

    # Split image name
    image_name = Path(img_path).stem
    image_suffix = Path(img_path).suffix
    # cv2.imwrite(str(output_dir_sauvola / image_name)+'-SauvolaWS='+str(window_size) + image_suffix, sauvola_image)
    # =============================================================================
    # INVERSE thresholding
    # =============================================================================
    th, sauvola_bin_image = cv2.threshold(
        sauvola_image, 0, 255, cv2.THRESH_BINARY_INV)

    img_denoised = remove_small_objects(sauvola_bin_image, denoised_size)

    original_name = image_name.split('pdpd')[0]

    isExist = os.path.exists(
        str(output_dir_denoised) + "/" + original_name)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(str(output_dir_denoised) + "/" + original_name)
        print("The new directory is created for " + original_name)

    print(str(output_dir_denoised) + "/" + original_name + "/" +
          str(image_name) + '-denoised-MANUAL' + image_suffix)
    cv2.imwrite(str(output_dir_denoised) + "/" + original_name + "/" +
                str(image_name) + '-denoised-MANUAL' + image_suffix, img_denoised)

    return img_denoised
