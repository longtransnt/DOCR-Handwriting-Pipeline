# DOCR-Handwriting-Pipeline

Front End User Interface for HANDWRITING RECOGNITION APPLICATION FOR TETANUS TREATMENT

## Acknowledgements

This project module has been developed and belongs to:

- [RMIT University Vietnam SGS](https://www.rmit.edu.vn/)
- [Oxford University Clinical Research Unit](https://www.oucru.org/)

## Authors

- [@longtransnt](https://github.com/longtransnt)
- [@chrisidenbui](https://github.com/chrisidenbui)
- [@julliannah](https://github.com/julliannah)
- [@s3681447](https://github.com/s3681447)

## Built With

This project has been built with:

[![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)]()

[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)]()

## Prerequisites

To install local or deployed application, the following steps should be done:

- Check if Windows version is 21H2. If not, update the Windows version.
- Install WSL2 (https://learn.microsoft.com/en-us/windows/wsl/install and https://releases.ubuntu.com/focal/)
- Install Annaconda on Ubuntu 20.04 (https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/)

[DEPLOY] To deploy this application on Amazon Web Services, you need to set up a G-Instance EC2.

- Prefered: g4dn.xlarge, 1, 4, 16, 1 x 125 NVMe SSD
- Install NVIDIA drivers on Ubuntu instance to run GPU (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)

[LOCAL] To run locally, you need to install CUDA Toolkit 11.3 (https://developer.nvidia.com/cuda-11.3.0-download-archive)

## Installations

First we need to update packages and install python3-pip:

```
sudo apt update
sudo apt install python3-pip
```

After that, install extra packages for installations:

```
sudo apt install build-essential
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

Then, install packages on anaconda environment, we need to use the env-spec.txt file:

```
pip install --user --upgrade aws-sam-cli
conda update --name base --file env-spec.txt
```

Detectron and Pytorch need to be installed from their distributions due to versioning:

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-cache-dir
```

## Environment Variables

To run this project, you will need to add the following environment variables to your ./Misc/Constant.py file:

`DEFAULT_PATH = [PATH TO THE PIPELINE]`

## Run pipeline's prediction flow in Folder mode

This project has two mode: Folder and Server

To run in Folder mode:

1. In folder "static/Input" and paste images/records into the folder

2. Command to activate the conda environment:

```
conda activate base
```

3. Command to run the pipeline:

```
python main.py -op Folder
```

3. Results at each stage can be founded in subfolders of "static/Output/":

```bash
 ├── PaperDetection                # Records that has irrelevant parts cropped out
 ├── Preprocessing                 # Normalized instances of the Paper Detected images
 ├── TextDetection                 # Cropped images of handwriting lines, divided in folders
 │   ├── .../coordinates.json      # Coordinates of each cropped images on the Paper Detected images
 ├── Adaptive                      # Adaptive Preprocessed images from the Text Detection instances
 │   ├── Adaptive-Preview          # Folder to store Preview manually processed images (for UI usage)
 │   ├── .../blur.json             # Blur degree of each processed image (for UI usage)
 └──  TextRecognition              # Translation of handwriting to machine text. Stored in json

```

## Run pipeline's prediction flow in Server mode

To run in local mode:

1. Command to activate the conda environment:

```
conda activate base
```

2. Command to run the pipeline locally in server mode:

```
python main.py -op Server
```

3. Command to run the pipeline:

```
python main.py -op Server
```

The services can be access by the following endpoints:

```
localhost:8080
```

4. Run the UI code and navigate to the following URL to upload a new record to input into the pipeline:

```
localhost:3000/input
```

5. Click on the image to start the operation in the UI
