import glob

import matplotlib.pyplot as plt

from vehicle_detection.features import FeatureExtractor
from vehicle_detection.util import load_image, get_filename

color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = False  # Spatial features on or off
spatial = 0  # Spatial binning dimension
hist_feat = False  # Histogram features on or off
hist_bins = 0  # Number of histogram bins
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

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')

files = glob.glob('output_images/training/*.png')
for file in files:
    filename = get_filename(file)
    image = load_image(file)
    features, hog_image = feature_extractor.get_hog_features_for_plotting(image)

    plt.imshow(hog_image)
    plt.savefig('output_images/hog/' + filename)
    plt.show()
