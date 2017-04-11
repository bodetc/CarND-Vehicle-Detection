import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import vehicle_detection.lesson_functions as lesson_functions
import vehicle_detection.util as util

images, labels = util.load_images('data')

# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 8
histbin = 32
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off

X = lesson_functions.extract_features(images, color_space='RGB', spatial_size=(spatial, spatial), hist_bins=histbin,
                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=False)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = labels

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:', spatial,'and', histbin, 'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
