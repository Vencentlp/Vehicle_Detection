# Vehicle Detection


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
  
### Import the libraries
```python
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip
%matplotlib inline
```
### Load training data
```python
car_dir_base = 'vehicles/'
noncar_dir_base = 'non_vehicles/'
cars = []
notcars = []
cars = glob.glob('vehicles/vehicles/**/*.png')
notcars = glob.glob('non-vehicles/non-vehicles/**/*.png')
print(' Number of cars images is',len(cars),'\n','Number of non car images is',len(notcars))
car_exam = mpimg.imread(cars[np.random.randint(0,len(cars))])
notcar_exam = mpimg.imread(notcars[np.random.randint(0,len(notcars))])
```
Number of cars images is 8792 
Number of non car images is 8968

Show examles of the dataset:
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/imge_example.jpg)

