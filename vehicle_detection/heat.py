import cv2
import numpy as np
from scipy.ndimage.measurements import label


class Heat:
    heatmap = None
    threshold_heatmap = None
    labels = None
    threshold = 2

    @staticmethod
    def __create_heatmap(image, bbox_list):
        heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def __apply_threshold(self):
        self.threshold_heatmap = np.copy(self.heatmap)
        # Zero out pixels below the threshold
        self.threshold_heatmap[self.threshold_heatmap <= self.threshold] = 0
        # Return thresholded map
        return self.threshold_heatmap

    def process_image(self, image, hot_windows):
        new_heatmap = self.__create_heatmap(image, hot_windows)

        if self.heatmap is None:
            self.heatmap = new_heatmap
        else:
            self.heatmap = .5 * self.heatmap + .5 * new_heatmap

        self.__apply_threshold()

        # Visualize the heatmap when displaying
        self.threshold_heatmap = np.clip(self.threshold_heatmap, 0, 255)

        # Find final boxes from heatmap using label function
        self.labels = label(self.threshold_heatmap)

        return self.labels

    def draw_labeled_bboxes(self, image):
        img = np.copy(image)
        # Iterate through all detected cars
        for car_number in range(1, self.labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img
