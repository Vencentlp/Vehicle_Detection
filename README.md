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

### Spatial Binning of Color
```python
# Define a function to compute binned color features
# Code from class
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
 ```
 The feature example
 ```python
 car_exam = cv2.imread(cars[np.random.randint(0,len(cars))])
notcar_exam = cv2.imread(notcars[np.random.randint(0,len(notcars))])
car_bin_vec = bin_spatial(car_exam)
notcar_bin_vec = bin_spatial(notcar_exam)
fig, axs = plt.subplots(2,2,figsize=(8,8))
axs = axs.ravel()
axs[0].axis('off')
axs[0].set_title('car', fontsize=10)
axs[0].imshow(car_exam)
axs[1].set_title('car_spin', fontsize=10)
axs[1].plot(car_bin_vec)
axs[2].axis('off')
axs[2].set_title('notcar', fontsize=10)
axs[2].imshow(notcar_exam)
axs[3].set_title('notcar_spin', fontsize=10)
axs[3].plot(notcar_bin_vec)
plt.savefig('output_images/bin_spatial.jpg')
 ```
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/bin_spatial.jpg)

### Histograms of Color
```python
# Define a function to compute color histogram features  
# Code from class
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins,range=(0,256))
    channel2_hist = np.histogram(img[:,:,1], bins=nbins,range=(0,256))
    channel3_hist = np.histogram(img[:,:,2], bins=nbins,range=(0,256))
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist,channel2_hist,channel3_hist,hist_features
```
Show example of Histograms of Color:
    
```python
car_ch1,car_ch2,car_ch3,car_hist = color_hist(car_exam)
notcar_ch1,notcar_ch2,notcar_ch3,notcar_hist = color_hist(notcar_exam)
bincent = (car_ch1[1][0:len(car_ch1[1])-1] + car_ch1[1][1:])/2
#print(notcar_ch1[0])
#print(bincent)
fig = plt.figure(figsize=(12,6))
plt.subplot(2,3,1)
plt.bar(bincent, car_ch1[0])
plt.xlim(0, 256)
plt.title('Car B Histogram')
plt.subplot(2,3,2)
plt.bar(bincent, car_ch2[0])
plt.xlim(0, 256)
plt.title('Car G Histogram')
plt.subplot(2,3,3)
plt.bar(bincent, car_ch3[0])
plt.xlim(0, 256)
plt.title('Car R Histogram')
plt.subplot(2,3,4)
plt.bar(bincent, notcar_ch1[0])
plt.xlim(0, 256)
plt.title('notCar B Histogram')
plt.subplot(2,3,5)
plt.bar(bincent, notcar_ch2[0])
plt.xlim(0, 256)
plt.title('notCar G Histogram')
plt.subplot(2,3,6)
plt.bar(bincent, notcar_ch3[0])
plt.xlim(0, 256)
plt.title('notCar R Histogram')
fig.tight_layout()
plt.savefig('output_images/color_hist.jpg')
```
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/color_hist.jpg)

### Extract HOG features
```python
# Define a function to return HOG features and visualization
# Code from class
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```
Show HOG feature examples:
```python
gray_car = cv2.cvtColor(car_exam, cv2.COLOR_RGB2GRAY)
gray_ncar = cv2.cvtColor(notcar_exam, cv2.COLOR_RGB2GRAY)
_, car_hog =  get_hog_features(gray_car, 9, 8, 2, vis=True, feature_vec = False)
_, ncar_hog =  get_hog_features(gray_ncar, 9, 8, 2, vis=True, feature_vec = False)
fig = plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.imshow(gray_car, cmap='gray')
plt.title('Example car image')
plt.subplot(2,2,2)
plt.imshow(car_hog, cmap='gray')
plt.title('HOG Visulization of example car')
plt.subplot(2,2,3)
plt.imshow(gray_ncar, cmap='gray')
plt.title('Example none car image')
plt.subplot(2,2,4)
plt.imshow(ncar_hog, cmap='gray')
plt.title('HOG Visulization of example not car')
fig.tight_layout()
plt.savefig('output_images/HOG.jpg')
```
