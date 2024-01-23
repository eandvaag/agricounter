# keras.json

```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```


# Supported models

- CenterNet
	+ Neck: ResNet Deconvolution
		- Backbone: ResNets: ResNet50, ResNet101, ResNet152

	+ Neck: EfficientDet BiFPN
		- Backbone: EfficientNets: D0, D1, D2, D3, D4, D5, D6, D7


- RetinaNet
	+ Neck: FPN
		- Backbone: ResNets: ResNet50, ResNet101, ResNet152


- YOLOv4
	+ Neck: SPP-PAN
		- Backbone: CSP-Darknet53




# Install

# get driver 515 installed
sudo ubuntu-drivers install nvidia-driver-515
# note using the software control panel in the gui seemed easier

# get python3.8
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8
sudo apt install python3.8-venv

# install cuda 11.2.2
sudo sh /home/erik/Downloads/cuda_11.2.2_460.32.03_linux.run --override

# install libcudnn8
sudo apt install ./libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo apt install ./libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb

# add lines below to ~/.bashrc
export PATH=$PATH:/usr/local/cuda-11.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64

# make venv
cd ~/Documents
python3.8 -m venv plant_detection/plant_detection_venv
source plant_detection/plant_detection_venv/bin/activate
/home/erik/Documents/plant_detection/plant_detection_venv/bin/python3.8 -m pip install --upgrade pip
pip3 install -r ./plant_detection/src/requirements.txt

# install gdal
sudo apt install libgdal-dev gdal-bin
sudo apt install python3.8-dev

# run gdalinfo --version, then get matching pygdal version
pip install pygdal==${matching_pygdal_version}