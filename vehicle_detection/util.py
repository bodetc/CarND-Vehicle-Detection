import glob

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def load_training_images(folder='data'):
    # Read in car and non-car images
    cars = []
    files = glob.glob(folder + '/vehicles/*/*.png')
    for file in files:
        cars.append(file)

    notcars = []
    files = glob.glob(folder + '/non-vehicles/*/*.png')
    for file in files:
        notcars.append(file)

    n_cars = len(cars)
    n_notcars = len(notcars)

    print('Loaded', n_cars, 'car and', n_notcars, 'not car images')

    images = np.hstack((cars, notcars))
    labels = np.hstack((np.ones(n_cars), np.zeros(n_notcars)))

    return images, labels


def load_test_images(folder='test_images'):
    images = []
    files = glob.glob(folder + '/*.jpg')
    for file in files:
        images.append(file)

    return images


def load_image(filename):
    # Read in each one by one
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_video(input_file, output_file, function):
    clip = VideoFileClip(input_file)
    output_clip = clip.fl_image(function)
    output_clip.write_videofile(output_file, audio=False)
