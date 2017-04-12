import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.features import FeatureExtractor
from vehicle_detection.lesson_functions import *
from vehicle_detection.util import *


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, feature_extractor):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = feature_extractor.extract_image_features(test_img)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = False  # Spatial features on or off
spatial = 8  # Spatial binning dimension
hist_feat = False  # Histogram features on or off
hist_bins = 32  # Number of histogram bins
hog_feat = True  # HOG features on or off
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

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
svc = LinearSVC()
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

for file in images:
    image = load_image(file)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, feature_extractor)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)

    plt.show()
