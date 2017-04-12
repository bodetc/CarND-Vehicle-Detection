import cv2
import numpy as np


class WindowSearch:
    window_sizes = [
        [48, [405, 495]],
        [64, [405, 495]],
        [96, [405, 540]],
        [128, [360, 540]],
        [160, [360, 540]],
        [192, [360, 630]],
        [256, [360, 630]],
        [384, [360, 630]],
    ]

    def __init__(self, classifier, scaler, feature_extractor):
        self.classifier = classifier
        self.scaler = scaler
        self.feature_extractor = feature_extractor

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    @staticmethod
    def __slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
                       xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def __search_windows(self, img, windows):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.feature_extractor.extract_image_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def search_image(self, image):
        all_windows = []

        for window_size in self.window_sizes:
            size = window_size[0]
            y_start_stop = window_size[1]
            windows = WindowSearch.__slide_window(image, y_start_stop=y_start_stop, xy_window=(size, size))
            all_windows.extend(windows)

        print('Searching a total of', len(all_windows), 'windows')

        hot_windows = self.__search_windows(image, all_windows)

        return hot_windows
