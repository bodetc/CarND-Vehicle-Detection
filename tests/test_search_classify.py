import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.features import FeatureExtractor
from vehicle_detection.lesson_functions import *
from vehicle_detection.util import *
from vehicle_detection.windows import WindowSearch


color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = True  # Spatial features on or off
spatial = 16  # Spatial binning dimension
hist_feat = True  # Histogram features on or off
hist_bins = 16  # Number of histogram bins
hog_feat = True  # HOG features on or off
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"

images, labels = load_training_images('data')

feature_extractor = FeatureExtractor(color_space=color_space,
                                     spatial_feat=spatial_feat, spatial_size=(spatial, spatial),
                                     hist_feat=hist_feat, hist_bins=hist_bins,
                                     hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, hog_channel=hog_channel)

X = feature_extractor.extract_features(images)

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

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC(C=0.1)
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
y_pred = svc.predict(X_test)
print('Test Accuracy of SVC = ', round(accuracy_score(y_test, y_pred), 4))
# Check the prediction time for a single sample
t = time.time()

images = load_test_images(folder='test_images')

window_search = WindowSearch(svc, X_scaler, feature_extractor)

for file in images:
    image = load_image(file)
    draw_image = np.copy(image)

    hot_windows = window_search.search_image(image)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)

    plt.show()
