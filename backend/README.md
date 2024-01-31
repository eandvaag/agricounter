# AgriCounter Python Process


## Install (Non-Docker)

Install NVIDIA driver 515
```
	sudo ubuntu-drivers install nvidia-driver-515
```
(Using the software control panel in the GUI may be easier.)



Install python3.8
```
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt install python3.8
	sudo apt install python3.8-venv
```


Install cuda 11.2.2
```
sudo sh ./cuda_11.2.2_460.32.03_linux.run --override
```


Install libcudnn8
```
	sudo apt install ./libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
	sudo apt install ./libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
```


Add lines below to `~/.bashrc`.
```
	export PATH=$PATH:/usr/local/cuda-11.2/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64
```


Make venv.
```
	python3.8 -m venv ./agricounter_venv
	source ./agricounter_venv/bin/activate
	./agricounter_venv/bin/python3.8 -m pip install --upgrade pip
	pip3 install -r ./backend/src/requirements.txt

```


Install gdal.
```
	sudo apt install libgdal-dev gdal-bin
	sudo apt install python3.8-dev
```


Run `gdalinfo --version`, then get matching pygdal version:
```
	pip install pygdal==${matching_pygdal_version}
```


Ensure `~/.keras/keras.json` contains the following:
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```


To start the Python process, execute the following command from the `backend/src` directory:
```
	python server.py
```