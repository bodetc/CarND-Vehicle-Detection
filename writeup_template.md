#Vehicle Detection Project

In this project, the goal is to establish a pipeline for the detection of vehicles on a video stream taken by a camera mounted on the dashboard of a car.
The use of a provided training set of labeled images allowed to train a linear SVM classifier.
To that end, a Histogram of Oriented Gradients (HOG) was combined with two other methods to extract a feature vector from the images.
The sliding-window technique was then used in combination with the trained classifier to search for cars in the full dashcam frame.
From there, after filtering for false positive and multiple detection, the bounding boxes of the detected cars are estimated and plotted on the original frame.

[//]: # (Image References)
[car]: output_images/training/445.png
[not_car]: output_images/training/extra8.png
[car_hog]: output_images/hog/445.png
[not_car_hog]: output_images/hog/extra8.png
[windows1]: ./output_images/windows/test1.jpg
[windows5]: ./output_images/windows/test5.jpg
[heatmap_5]: ./output_images/heat/heatmap_test5.jpg
[threshold_5]: ./output_images/heat/threshold_heatmap_test5.jpg
[bbox_5]: ./output_images/heat/bbox_test5.jpg


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

The corresponding code is contained in the method `__bin_spatial` in the file `vehicle_detection/features.py`.  

#### Color histogram features

The second approach consist into taking histograms of the values of each color channel, summed over all pixels of the image.
By taking `N` bins for the histogram, the number of feature becomes `3N`.
One should note that with this method, all the structure information is lost.
However, combined with other methods of feature extraction, using histograms can improve accuracy.

By taking 32 bins in the LUV colorspace, one obtains 96 features.
By testing on the color histogram alone, the testing accuracy is of 91.9 percent.

The corresponding code is contained in the method `__color_hist` in the file `vehicle_detection/features.py`.  

#### Histogram of Oriented Gradients (HOG) feature extraction

The Histogram of Oriented gradient is a method used for object detection.
In effect, presents some representation of the shape of an object in an image by calculating the gradient for each color channel in the x and y direction and then by putting the gradient orientation in an histogram.
The algorithm is implemented in `skimage.hog()`.

Here two examples of HOG transformations in the `LUV` color space using only the first channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`.
The first image is the car example see above, while the second image is the non-car example.

![alt text][car_hog]

![alt text][not_car_hog]

With those parameters, the feature vector length is 576 and the model has a testing accuracy of 94.5 percent.
While the parameters were generally chosen for the best accuracy of the classifier, the number of cell per block was maintained at (1, 1).
Extending this to (2, 2) does increase the accuracy of the classifier, but it also drastically slows down the speed of the feature computation.

The code for this step is contained in the method `__get_hog_features` in the file `vehicle_detection/features.py`.  

#### Feature scaling

As the features extracted from different methods have different scaling, it is important to normalize the final feature vector to avoid having some feature strongly overweighting other.
The `StandartScaler` provided by `sklearn` was used to provide a normalized feature vector.

### Model and training

Using the provided dataset, the images are first loaded and then transformed into feature vectors.
Then, the feature vector is scaled using the `StandartScaler`.

To prepare for training, the dataset is then split into a training and a testing set using `train_test_split` from `sklearn`.
Twenty percent of the dataset is used for testing.

The model itself was chosen as a linear SVM, as its performance is already sufficient for the purpose of this project.
The model is then trained on the training set and its accuracy calculated on the testing set.

The code for this section can be found in the file `vehicle_detection/training.py`.

### Parameters

The final parameters for feature extraction where chosen to maximize the test accuracy of the linear SVM model on the testing data.
The respective size of each feature vector and the computation time was also taken into account.
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

![alt text][windows5]

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

## Video Pipeline

The video pipeline is nearly identical to the picture pipeline.
The only difference is the inclusion of memory in the heatmap.
The new heatmap is averaged with the previously calculated heatmap before applying thresholding and calculating labels.

`self.heatmap = .5 * self.heatmap + .5 * new_heatmap`

As this is done for each frame, the old heatmaps are exponentially decaying.
This provides additional stability towards false positive, as they happen only on some frames.

The pipeline is called by the last line of `vehicle_detection.py`.

##Discussion

Here's a [link to my video result](./tests/project_video.mp4).
As one can see, the model successfully manages to identify the cars driving nearby.
There is a very limited number of false positive, those are usually triggered by cars driving in the opposite direction.

The bounding boxes are a bit jumpy. This is expected due to the discrete amount of sliding windows, which implies a limited amount of possible positions for bounding boxes.

While this pipeline seems quite robust at identifying what is a car and where there, but it tends underestimate the size of the car.
This effect can be reduced by lowering the heatmap threshold from 3 to 2, but this will also increase the number of false positives.

Another solution might be to average not only the heatmap over several frames, but also the resulting bounding boxes.
However, the resolution will always be limited by the discrete amount of search windows.
It might be a better solution to merge the result of this pipeline with the output of other sensors which are better suited
for the task, such as radar or lidar.
The video pipeline would identify what are cars, and the other sensors would provide their precise location.

Furthermore, the model was only trained using daytime images of cars.
This means that the model might not be able to recognise cars in other situation, such as nighttime.
It would also not be able to recognise other kind of objects that might be present on the road, such as motorbike, trucks or pedestrians.
