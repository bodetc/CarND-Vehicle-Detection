##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/445.png
[not_car]: ./output_images/extra8.png
[windows1]: ./output_images/windows/test1.jpg
[windows5]: ./output_images/windows/test5.jpg
[heatmap_5]: ./output_images/heat/heatmap_test5.jpg
[threshold_5]: ./output_images/heat/threshold_heatmap_test5.jpg
[bbox_5]: ./output_images/heat/bbox_test5.jpg

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


## Classifier

In this section, I will present how a classifier was created to distinguish car from non-car images.
This classifier will be used later on multiple windows (sub-images) of the camera frames to decide whether this window contains a car or not.

For training, a dataset of labeled car and not car images of size 64x64.
The set contains 8792 car and 8968 not car images. Below are two sample images from this dataset:

![alt text][car]
![alt text][not_car]

###Feature Extraction

The first step is to transform the input images into a series of features to be used as an input to the classifier.

#### Colormap features

The first and most simple method to create features is to simply use color channel of each pixel as features.
This method is very prone to overfitting and may have poor generalisation characteristics.

However, by reducing the resolution of the image and by choosing an appropriate color space,
one can still obtain decent classification result.

By choosing a resolution of 8x8 pixels and the LUV color space, we genereate a feature vector of length 192.
Training our model on the colormap features alone gives a testing accuracy of 87.9%

The corresponding code is contained in the method `__bin_spatial` in the file `vehicle_detection/features.py`).  

#### Color histogram features

The second approach consist into taking histograms of the values of each color channel, summed over all pixels of the image.
By taking `N` bins for the histogram, the number of feature becomes `3N`.
One should note that with this method, all the structure information is lost.
However, combined with other methods of feature extraction, using histograms can improve accuracy.

By taking 32 bins in the LUV colorspace, one obtains 96 features.
By testing on the color histogram alone, the testing accuracy is of 91.9 percent.

The corresponding code is contained in the method `__color_hist` in the file `vehicle_detection/features.py`).  

#### Histogram of Oriented Gradients (HOG) feature extraction

The code for this step is contained in the method `__get_hog_features` in the file `vehicle_detection/features.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### Feature scaling

### Model and training

Once 

### Parameters

The final parameters for feature extraction where chosen to maximize the test accuracy of the linear SVM model on the testing data.
The final parameters for each feature extraction method is presented above.

Furthermore, the linear SVM model also has a `C` meta-parameter. This parameter can be used to prevent overfitting of the data.
It this case, a value of `C=0.001` was manually found to be a good value.

With those parameters, the feature vector has a length of 1392 and the testing accuracy is 98.5%.

##Sliding Window Search

In order to find cars in the image, it is needed to analyze sub-section of the image separately.
This is done by creating small windows next to each other (eventually with overlap) and sending the cropped image to our classifier.

Those search windows are generated programmatically by giving a desired size for the window, the region of the image to search, and the overlap between one window and the next.
The corresponding code can be found in the method `__slide_window` in the file `vehicle_detection/windows.py`). 

A compromise has to be found between the number of search windows (with affects the calculation speed) and the accuracy of the bounding boxes of the car.
The more windows are searched, the more likely it is to be able to be able to filter false positive and create accurate bounding boxes for the car.
However, this increases the computation time accordingly.

Restricting the area in which to generate the window can also improve the calculation time. This is done by not searching areas where the presence of a car is unlikely (i.e. the sky).

The following table shows the size of the sliding windows, and where in the image they were generated.
The overlap betweend windows is chosen at 50% in both directions.
This creates a total of 343 windows to search.

|  `y`-search  |    Window size                     |
| :----------: | ---------------------------------- |
|  405 to 495  | 48x48, 64x64                       |
|  405 to 540  | 80x80, 96x96                       |
|  360 to 540  | 112x112, 128x128, 144x144, 160x160 |
|  360 to 600  | 192x192                            |
|  360 to 630  | 256x256                            |

The cropped image contained those search windows is then send to the classifier, and only the windows which are predicted to contain a car are retained.
Here is a few sample image with the search windows containing a car drawn.

![alt text][windows1]

![alt text][windows3]

## Heatmap and bounding boxes

As it can be noticed from the sample pictures of the previous section,
the same car is usually detected multiple times by overlapping windows.
There are also some false positives, i.e. selected windows not containing cars.
Sometimes, cars from the opposite direction are detected as well, but this is not very reliable as the cars are further away and are partially masked by the separator and vegetation.

In order to consolidate the multiple detections and to filter out false positive, and heatmap is implemented.
The heatmap is a
Here is an example:
![alt text][heatmap_5]

Then, false positive are filtered by removing all pixels below a given threshold.
In practice, we used a filter of 3 on static images, meaning that only pixel that are detected at least in three windows as cars will be selected.
The resulting heatmap looks like this:
![alt text][threshold_5]

Finally, labels (bounding boxes) are calculated on the thresholded heatmap, and the bounding boxes are plotted on the original image.
The final result looks like this:
![alt text][bbox_5]

The code for heatmap calculation can be found in `vehicle_detection/heat.py`

## Video Implementation

### Pipeline

The video pipeline is nearly identical to the picture pipeline.
The only difference is the inclusion of memory in the heatmap.
The new heatmap is averaged with the previously calculated heatmap before applying thresholding and calculating labels.

`self.heatmap = .5 * self.heatmap + .5 * new_heatmap`

As this is done for each frame, the old heatmaps are exponentially decaying.
This provides additional stability towards false positive, as they happen only on some frames.

The pipeline is called by the last line of `vehicle_detection.py`.

### Video output

Here's a [link to my video result](./tests/project_video.mp4).

##Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

