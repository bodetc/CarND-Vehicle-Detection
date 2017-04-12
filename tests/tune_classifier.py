from sklearn.svm import LinearSVC

import vehicle_detection.training
from vehicle_detection.features import FeatureExtractor

color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = True  # Spatial features on or off
spatial = 8  # Spatial binning dimension
hist_feat = True  # Histogram features on or off
hist_bins = 32  # Number of histogram bins
hog_feat = True  # HOG features on or off
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"

feature_extractor = FeatureExtractor(color_space=color_space,
                                     spatial_feat=spatial_feat, spatial_size=(spatial, spatial),
                                     hist_feat=hist_feat, hist_bins=hist_bins,
                                     hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, hog_channel=hog_channel)

print('Using spatial binning of:', spatial, 'and', hist_bins, 'histogram bins')
print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')

# Use a linear SVC
model = LinearSVC()
scaler = vehicle_detection.training.prepare_and_train_model(model, feature_extractor)
