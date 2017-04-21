import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from vehicle_detection.features import FeatureExtractor
from vehicle_detection.heat import Heat
from vehicle_detection.lesson_functions import *
from vehicle_detection.training import prepare_and_train_model
from vehicle_detection.util import *
from vehicle_detection.windows import WindowSearch

color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_feat = True  # Spatial features on or off
spatial = 16  # Spatial binning dimension
hist_feat = True  # Histogram features on or off
hist_bins = 16  # Number of histogram bins
hog_feat = True  # HOG features on or off
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 1  # HOG cells per block
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
model = LinearSVC(C=0.001)
scaler = prepare_and_train_model(model, feature_extractor)

window_search = WindowSearch(model, scaler, feature_extractor)


def process_image_boxes(image):
    draw_image = np.copy(image)
    hot_windows = window_search.search_image(image)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return window_img


heat = Heat()


def process_image_heatmap(image):
    hot_windows = window_search.search_image(image)
    heat.process_image(image, hot_windows)
    draw_img = heat.draw_labeled_bboxes(image)
    return draw_img


images = load_test_images(folder='test_images')
for file in images:
    image = load_image(file)

    heat = Heat()
    heat.threshold = 3
    window_img = process_image_heatmap(image)
    plt.imshow(window_img)

    plt.show()

heat = Heat()
process_video('project_video.mp4', 'tests/project_video.mp4', process_image_heatmap)
