# FPRG 2023 Repos Project 03 - Face Reconstruct

## License

The code of HSEmotion Python Library is released under the Apache-2.0 License. There is no limitation for both academic and commercial usage.

## Installing

Tested with Ubuntu 22.04. GPU: RTX3090 24GBVRAM

Use environment.yml to generate conda environment with

```
conda env create --file environment.yml -n FaceRecon
```


## Usage for FPRG Project 03
### Train
The ```train_64.py``` reads the files of a given folder pair of original and occluded face images, resizes them to 64x64 Pixel, saves this image files as one .pkl file and starts the training.

### Test / Inference
The ```test_64.py``` reads the image files as resized .pkl files and uses the occluded one to generate reconstructed images. Which images can be defined in the code. All generated images will be saved as .pkl file. To make images out of the .pkl file use ```write_pkl_file_to_jpgs.py```.

## 
 

based on

https://github.com/ahmetmeleq/Face-Completion---Occlusion-Restoration-GAN

