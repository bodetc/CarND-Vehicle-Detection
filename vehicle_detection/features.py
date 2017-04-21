# noinspection PyUnresolvedReferences
import better_exceptions
import cv2
import numpy as np
from skimage.feature import hog

from vehicle_detection.util import load_image


class FeatureExtractor:
    def __init__(self,
                 color_space='RGB',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                 orient=9,  # HOG orientations
                 pix_per_cell=8,  # HOG pixels per cell
                 cell_per_block=2,  # HOG cells per block
                 hog_channel=0,  # Can be 0, 1, 2, or "ALL"
                 spatial_size=(16, 16),  # Spatial binning dimensions
                 hist_bins=16,  # Number of histogram bins
                 spatial_feat=False,  # Spatial features on or off
                 hist_feat=False,  # Histogram features on or off
                 hog_feat=False,  # HOG features on or off
                 ):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    # Define a function to return HOG features and visualization
    def __get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def get_hog_features_for_plotting(self, img, channel=0):
        return self.__get_hog_features(img[:, :, channel], vis=True, feature_vec=True)

    # Define a function to compute binned color features
    def __bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def __color_hist(self, img, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def extract_image_features(self, image):
        file_features = []
        # apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if self.spatial_feat:
            spatial_features = self.__bin_spatial(feature_image)
            file_features.append(spatial_features)
        if self.hist_feat:
            # Apply color_hist()
            hist_features = self.__color_hist(feature_image)
            file_features.append(hist_features)
        if self.hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(
                        self.__get_hog_features(feature_image[:, :, channel], vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.__get_hog_features(feature_image[:, :, self.hog_channel], vis=False,
                                                       feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        return np.concatenate(file_features)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = load_image(file)

            file_features = self.extract_image_features(image)
            features.append(file_features)
        # Return list of feature vectors
        return features
