**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/imge_example.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/HOG.jpg)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided to use all channels of 'YUV' colorsapce and 'orentation =9, pixls_per_cell=(8,8)' and 'cells_per_block=(2,2)'.The following table are all the parameter combinations I explored. I firstly consider the speed with accuracy taken into consideration. There exists a trade off to determine the final parameters. If the calculation speed is short the accuracy will be lower. So I decided to use the first row of parameters to reduce the false positives and shorter time to calculate.

| Configuration label | Colorspace | Orientations | Pixles Per Cell | Cells Per Block |Hog Channel |Extract Time |Accuracy |Train Time |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|    
| 1      | YUV        | 9 | 8 | 2 | ALL | 149.32 | 0.9831 | 27.79 |
| 2      | YCrCb      | 9 | 8 | 2 | ALL | 183.14 | 0.9809 | 29.85 |
| 3      | RGB        | 9 | 8 | 2 | ALL | 190.05 | 0.9662 | 48.64 |
| 4      | HSV        | 9 | 8 | 2 | ALL | 175.42 | 0.9817 | 34.04 |
| 5      | HLS        | 9 | 8 | 2 | ALL | 182.96 | 0.9794 | 36.26 |
| 6      | YUV        | 9 | 8 | 2 | 1 | 76.5 | 0.9685 | 21.44 |
| 7      | YUV        | 9 | 8 | 2 | 2 | 83.85 | 0.9648 | 21.09 |
| 8      | YUV        | 12 | 8 | 2 | 1 | 79.54 | 0.969 | 23.02 |
| 9      | YUV        | 7 | 8 | 2 | 1 | 75.61 | 0.9657 | 18.83 |
| 10      | YUV        | 5 | 8 | 2 | 1 | 90.25 | 0.9648 | 16.84 |
| 11      | YUV        | 7 | 16 | 2 | 1 | 106.36 | 0.9657 | 19.23 |
| 12      | YUV        | 7 | 4 | 2 | 1 | 329.37 | 0.9628 | 46.9 |
| 13      | YUV        | 7 | 8 | 1 | 1 | 176.13 | 0.9595 | 22.34 |
| 14      | YUV        | 7 | 8 | 3 | 1 | 193.77 | 0.9662 | 26.37 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the hog features,  histogram of color and spatial binning of color features. The code can be foound in 29th code cell.Before tarning, I extracted features for cars and not cars images. Then I split the dataset into 2 parts:train dataset and test dataset. Then I normalize all the features. Finally, I calculated the accuracy for the test dataset.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;). The basic rules to determine the size of sliding windows is: longer the distance, smaller the window size. So i choosed smaller window size near the middle of the image and larger window size near the bottom the image. The following images shows the windows size.

Image with window size-64
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/boxes_with64size.jpg)

Image with window size-96:
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/boxeswith96size.jpg)

Image with window size-128:
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/boxeswith128size.jpg)

For the overlap of windows, I want to detect cars' shape in much more accuracy so I choose high overlap of 75%.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/0carpos.jpg)
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/1carpos.jpg)
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/2carpos.jpg)
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/3carpos.jpg)
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/4carpos.jpg)
![alt text](https://github.com/Vencentlp/Vehicle_Detection/raw/master/output_images/5carpos.jpg)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/Vencentlp/Vehicle_Detection/blob/master/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame and the corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

