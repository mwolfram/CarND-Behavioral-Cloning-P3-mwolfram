import csv
import cv2
import numpy as np

# TODO random seed

# read csv data
lines = []
with open('../CarND-Behavioral-Cloning-P3-data/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  iterlines = iter(reader)
  next(iterlines)
  for line in iterlines:
    lines.append(line)

# read images and measurements
images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = '../CarND-Behavioral-Cloning-P3-data/data/IMG/' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

# convert images and measurements to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# construct a basic network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

# Simple model
#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

# LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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

# split off validation data, shuffle, train the network
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

# save the resulting model
model.save('model.h5')



