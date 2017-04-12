import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC

from vehicle_detection.features import FeatureExtractor
from vehicle_detection.lesson_functions import *
from vehicle_detection.training import prepare_and_train_model
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

feature_extractor = FeatureExtractor(color_space=color_space,
                                     spatial_feat=spatial_feat, spatial_size=(spatial, spatial),
                                     hist_feat=hist_feat, hist_bins=hist_bins,
                                     hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, hog_channel=hog_channel)

print('Using spatial binning of:', spatial, 'and', hist_bins, 'histogram bins')
print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')

# Use a linear SVC
model = LinearSVC(C=0.1)
scaler = prepare_and_train_model(model, feature_extractor)

window_search = WindowSearch(model, scaler, feature_extractor)


def process_image_boxes(image):
    draw_image = np.copy(image)
    hot_windows = window_search.search_image(image)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return window_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


heat = None


def process_image_heatmap(image):
    hot_windows = window_search.search_image(image)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img


images = load_test_images(folder='test_images')
for file in images:
    image = load_image(file)

    window_img = process_image_heatmap(image)
    plt.imshow(window_img)

    plt.show()

# process_video('project_video.mp4', 'tests/project_video.mp4', process_image_boxes)
