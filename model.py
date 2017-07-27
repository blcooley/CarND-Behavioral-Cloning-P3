import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

samples = []

"""
There are several directories of images and csv files that contain the generated
training data. The directories are in the format "IMG_<extension>" with corresponding csv
file labeled "driving_log_<extension>". First we will load all the csv files by looping
through the extensions. Note that extension '8' corresponds to the training data
provided by Udacity.
"""
for extension_string in ['forward', 'backward', 'latest','4', '5', '6', '7', '8']:
    with open("driving_log_" + extension_string + ".csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # We're going to save a tuple including the extension string because
            # we need it later to load the file
            samples.append((line, extension_string))

# Now let's split the data into training and validation samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)

# Build a generator function patterned after the generator in lecture 17 of lesson 8
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for (batch_sample, extension_string) in batch_samples:
                # We want to include side camera images, so we need the first three
                # entries of the line, and we need a correction to apply to the
                # steering measurements for the side images
                for (file, steer_correction) in zip(batch_sample[:3], [0, 0.3, -0.3]):
                    filename = file.split('/')[-1]
                    image = cv2.imread("IMG_" + extension_string + "/" + filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    measurement = float(batch_sample[3]) + steer_correction
                    measurements.append(measurement)
                    # For each image and measurement, we will augment the data
                    # by flipping the image left-to-right (doubling data)
                    images.append(np.fliplr(image))
                    measurements.append(-measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

"""
We'll build the model below. This is the NVIDIA model structure discussed
in lecture 14, but with the addition of droupouts for three of the final
Dense layers to fight overfitting
"""
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

"""
Now we'll fit the model using the generator. Note that for each entry in
train_samples or validation_samples, we actually have 6 images. There are
the three images (center, left, right), plus the augmented data images we
get by flipping the images left-to-right. This accounts for the '*6' term
in the samples_per_epoch and nb_val_samples parameters in the function call
"""
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples)*6, nb_epoch=3)

model.save('model.h5')
exit()

