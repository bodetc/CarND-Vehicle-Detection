import glob
import numpy as np

def load_images(folder = 'data'):
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