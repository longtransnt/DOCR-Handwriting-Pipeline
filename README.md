# OUCRU-Paper-Detection
## Requirement
- **Prefer Linux or WSL2 on Windows**. Install WSL2 and Ubuntu 20.04 on Windows (*Require Some Hard Drive*) https://docs.microsoft.com/en-us/windows/wsl/install
- Make sure that your Windows version is 21H2. If not, CUDA driver will not register with WSL2, In order to update to 21H2, look at this link https://appuals.com/install-windows-10-version-21h2/
- MiniConda or Anaconda install in Linux. Please follow this guide but replace package with version 2021.11 https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/
- Create an environment in conda and install all required library by:
    	- Building identical conda environments (Linux, WSL2 only): [spec-file](/spec-file.txt) is provided in the repo. https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments
	- Run pip install requirements.txt
- **Detectron2 installation with/without GPU**.
    - Go to this link https://detectron2.readthedocs.io/en/latest/tutorials/install.html for information if you use different versions. You can skip reading and following the next lines,
	- *With GPU*: run 
    ```
    python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
    ```
	- *Without GPU*: run 
    ```
    python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
    ```
-  **Set up X-server in order to run GUI applications**
	- VcXsrv installation with: https://sourceforge.net/projects/vcxsrv/
	- Run XLaunch and input configuration in Ubuntu after finishing installations follow this video configuration: https://www.youtube.com/watch?v=4SZXbl9KVsw&t=183s
(NOTE: Please download models **weights** folder from [Drive](https://drive.google.com/drive/folders/12RvrCU0ZVf7UfoBRlVD2UvG5H2xiebjs?usp=sharing) and put **"weights"** directory to [Paper Detection's directory](/Paper_Detection/weights))

## Run the application
- In order to annotate images with ROI selection: Run 
    ```
    python MaskRCNN.py
    ```