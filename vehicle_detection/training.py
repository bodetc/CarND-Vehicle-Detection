import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from vehicle_detection.util import load_training_images


def train_model(model, X_train, y_train, X_test, y_test):
    # Check the training time for the SVC
    t = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    y_pred = model.predict(X_test)
    print('Test Accuracy of SVC = ', round(accuracy_score(y_test, y_pred), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', model.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


def prepare_and_train_model(model, feature_extractor, training_data_folder='data'):
    images, labels = load_training_images(training_data_folder)

    features = feature_extractor.extract_features(images)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(features)

    # Define the labels vector
    y = labels

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))

    train_model(model, X_train, y_train, X_test, y_test)
