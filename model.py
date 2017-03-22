import csv
import cv2
import numpy as np
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split

# TODO random seed

# configuration
SIDE_IMAGE_STEERING_BIAS = 0.2
DATA_FOLDER = "../CarND-Behavioral-Cloning-P3-data/data/"
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32

def getCSVLines():
  lines = []
  with open(DATA_FOLDER + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    iterlines = iter(reader)
    next(iterlines)
    for line in iterlines:
      lines.append(line)
  return lines

# reads images and measurements from samples (csv file lines)
def readImagesAndMeasurements(samples, augment=True):
  images = []
  measurements = []
  for line in samples:
    for i in range(3):
      source_path = line[i]
      filename = source_path.split('/')[-1]
      current_path = DATA_FOLDER + "IMG/" + filename
      image = cv2.imread(current_path)
      images.append(image)
      measurement = float(line[3])
      if i == 0:
        # center image
        measurements.append(measurement)
      elif i == 1:
        # left image
        measurements.append(measurement + SIDE_IMAGE_STEERING_BIAS)
      elif i == 2:
        # right image
        measurements.append(measurement - SIDE_IMAGE_STEERING_BIAS)

  if augment:
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
      augmented_images.append(image)
      augmented_measurements.append(measurement)
      augmented_images.append(cv2.flip(image, 1))
      augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements

  return images, measurements

def generator(samples, batch_size=BATCH_SIZE):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images, angles = readImagesAndMeasurements(batch_samples)

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

# read csv data
samples = getCSVLines()

# create generators for training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=VALIDATION_SPLIT)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# read images and measurements
#images, measurements = readImagesAndMeasurements(samples)

# convert images and measurements to numpy arrays, use original data
#X_train = np.array(images)
#y_train = np.array(measurements)

# construct a basic network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D

# Simple model
#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

# LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# TODO preprocessing layers 1x1
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# choose loss function and optimizer, compile the model
model.compile(loss='mse', optimizer='adam')

# fit model
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

# fit model with generator
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

# save the resulting model
model.save('model.h5')



