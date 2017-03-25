import random
import numpy as np
import sys

# set a fixed random seed
random.seed(1337)
np.random.seed(1337)

import csv
import cv2
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import zipfile
import matplotlib as mpl
mpl.use('Agg') # run headless
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout
from keras.regularizers import l2

# configuration
USE_FLOYD = False
DO_TRAIN = False
SIDE_IMAGE_STEERING_BIAS = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
PREDICT_IMAGES = ["sample_data/center_2016_12_01_13_31_13_177.jpg", "t2_forward_data/center_2017_03_22_17_03_16_764.jpg"]

if USE_FLOYD:
  # running on floydhub
  DATA_FOLDER = "/input/" 
  DATASETS = ["t1_reverse_data.csv", "t1_udacity_data.csv", "t2_forward_data.csv"]
  EPOCHS = 7
  OUTPUT_FOLDER = "/output/"
else:
  # running on local machine
  DATA_FOLDER = "../CarND-Behavioral-Cloning-P3-data/multiple_data/" 
  DATASETS = ["sample_data.csv"]
  EPOCHS = 2
  OUTPUT_FOLDER = ""

# =========================================
# Toolkit

# reads lines from a csv file, skipping the first line
def getCSVLines(csvpath):
  lines = []
  with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    iterlines = iter(reader)
    next(iterlines)
    for line in iterlines:
      lines.append(line)
  return lines

# reads lines from all given datasets and concatenates them
def getCSVLinesFromDatasets(data_folder, datasets):
  lines = []
  for entry in datasets:
    lines.extend(getCSVLines(data_folder + entry))
  return lines

# reads an image from a zip file
def getImageFromZip(zipname, filename):
  zipped_images = zipfile.ZipFile(DATA_FOLDER + zipname + ".zip")
  imagedata = zipped_images.read(filename)
  image = cv2.imdecode(np.frombuffer(imagedata, np.uint8), 1)
  return image

# reads images and measurements from samples (csv file lines)
def readImagesAndMeasurements(samples, augment=True):
  images = []
  measurements = []
  zipcache = {}
  for line in samples:
    for i in range(3):
      source_path = line[i]
      filename = source_path.split('/')[-1]
      zipname = source_path.split('/')[-2].strip()
      
      if zipname not in zipcache.keys():
        zipcache[zipname] = zipfile.ZipFile(DATA_FOLDER + zipname + ".zip")

      zipped_images = zipcache[zipname]
      imagedata = zipped_images.read(filename)
      image = cv2.imdecode(np.frombuffer(imagedata, np.uint8), 1)
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

# generates shuffled data from given samples
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

def getBasicModel():
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((70, 25), (0, 0))))
  model.add(Flatten())
  model.add(Dense(1))

  # choose loss function and optimizer, compile the model
  model.compile(loss='mse', optimizer='adam')

  return model

def getLeNetModel():
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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
  return model

def getNVIDIAModel():
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((70, 25), (0, 0))))
  model.add(Convolution2D(10, 1, 1, activation="relu", W_regularizer=l2(0.01))) # let the model choose the best color space
  model.add(Dropout(0.5))
  model.add(Convolution2D(3 , 1, 1, activation="relu", W_regularizer=l2(0.01))) # --"--
  model.add(Dropout(0.5))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.01)))
  model.add(Flatten())
  model.add(Dense(100, W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(50, W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(1, W_regularizer=l2(0.01)))

  # choose loss function and optimizer, compile the model
  model.compile(loss='mse', optimizer='adam')
  return model

def getPartialNVIDIAModelAndSetWeights(original_model_with_weights, depth=0):
  model = original_model_with_weights

  # Construct a second model for visualizing the activation
  model2 = Sequential()
  model2.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  model2.add(Cropping2D(cropping=((70, 25), (0, 0))))
  model2.add(Convolution2D(10, 1, 1, activation="relu", W_regularizer=l2(0.01), weights=model.layers[2].get_weights())) # let the model choose the best color space
  if depth <= 0: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(3 , 1, 1, activation="relu", W_regularizer=l2(0.01), weights=model.layers[4].get_weights())) # --"--
  if depth is 1: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01), weights=model.layers[6].get_weights()))
  if depth is 2: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01), weights=model.layers[8].get_weights()))
  if depth is 3: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation="relu", W_regularizer=l2(0.01), weights=model.layers[10].get_weights()))
  if depth is 4: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.01), weights=model.layers[12].get_weights()))
  if depth is 5: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Dropout(0.5))
  model2.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.01), weights=model.layers[14].get_weights()))
  if depth >= 6: 
    model2.compile(loss='mse', optimizer='adam')
    return model2
  model2.add(Flatten())
  model2.add(Dense(100, W_regularizer=l2(0.01)))
  model2.add(Dropout(0.5))
  model2.add(Dense(50, W_regularizer=l2(0.01)))
  model2.add(Dropout(0.5))
  model2.add(Dense(10, W_regularizer=l2(0.01)))
  model2.add(Dropout(0.5))
  model2.add(Dense(1, W_regularizer=l2(0.01)))

  model2.compile(loss='mse', optimizer='adam')
  return model2

def saveHistoryPlot(history_object):
  plt.plot(history_object.history['loss'])
  plt.plot(history_object.history['val_loss'])
  plt.title('model mean squared error loss')
  plt.ylabel('mean squared error loss')
  plt.xlabel('epoch')
  plt.legend(['training set', 'validation set'], loc='upper right')
  plt.show()

  plt.savefig(OUTPUT_FOLDER + 'history.png')

  plt.cla()   # Clear axis
  plt.clf()   # Clear figure
  plt.close() # Close a figure window

def train_model(model, samples):
  # create generators for training and validation sets
  train_samples, validation_samples = train_test_split(samples, test_size=VALIDATION_SPLIT)
  train_generator = generator(train_samples, batch_size=BATCH_SIZE)
  validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

  # fit model with generator
  history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)
  
  return history_object

# visualizes activation maps and saves these images under target_file
def outputFeatureMap(activation, target_file, activation_min=-1, activation_max=-1 ,plt_num=1):
  featuremaps = activation.shape[3]
  fig = plt.figure(plt_num, figsize=(15,15))
  for featuremap in range(featuremaps):
    plot = fig.add_subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
    #plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
    if activation_min != -1 & activation_max != -1:
      plot.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
    elif activation_max != -1:
      plot.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
    elif activation_min !=-1:
      plot.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
    else:
      plot.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
  fig.savefig(OUTPUT_FOLDER + target_file)
  plt.cla()   # Clear axis
  plt.clf()   # Clear figure
  plt.close() # Close a figure window
  fig.clf()

def getActivationMapFromImage(model, dataset_name, filename):
  image_array = np.asarray(getImageFromZip(dataset_name, filename))
  activation = model.predict(image_array[None, :, :, :], batch_size=1)
  return activation

# generates an activation map for each image in the list.
# an image is addressed as <dataset_name>/<filename>
# the activation maps are generated for several network depths
def generateActivationMapsForImages(images, depths, target_folder="feature_maps/"):

  for image in images:
    dataset_name = image.split('/')[0].strip()
    filename = image.split('/')[1].strip()

    original_image = getImageFromZip(dataset_name, filename)
    cv2.imwrite(OUTPUT_FOLDER + target_folder + dataset_name + "_" + filename + "_original.png", original_image)

    for depth in range(depths):
      # get a partial model with the weights set from the original model
      partial_model = getPartialNVIDIAModelAndSetWeights(model, depth)

      # get activation map from one image
      activation = getActivationMapFromImage(partial_model, dataset_name, filename)

      # save the activation as image
      outputFeatureMap(activation, target_folder + dataset_name + "_" + filename + "_" + str(depth) + "_features.png")


# =========================================
# Execute experiment

# read csv data
samples = getCSVLinesFromDatasets(DATA_FOLDER, DATASETS)

# Create NVIDIA model
model = getNVIDIAModel()

if DO_TRAIN:
  # train model
  history_object = train_model(model, samples)

  # save trained model
  model.save(OUTPUT_FOLDER + 'model.h5')
  
  # plot the history and save as png
  saveHistoryPlot(history_object)

else: 
  # load previous weights
  model.load_weights('model.h5')

# generate activation maps for images
generateActivationMapsForImages(PREDICT_IMAGES, 5)

