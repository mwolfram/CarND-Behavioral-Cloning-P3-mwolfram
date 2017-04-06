# CarND-Behavioral-Cloning-P3-mwolfram

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_driving_left]: ./writeup_images/left_2016_12_01_13_30_48_287.jpg "Center Driving, left camera"
[center_driving_center]: ./writeup_images/center_2016_12_01_13_30_48_287.jpg "Center Driving, center camera"
[center_driving_right]: ./writeup_images/right_2016_12_01_13_30_48_287.jpg "Center Driving, right camera"
[current_model_history]: ./history.png "Current Model Training History"
[history_with_l2]: ./writeup_images/history_with_l2.png "Training History with L2 regularization"
[sample_track2]: ./feature_maps/t2_forward_data_center_2017_03_22_17_03_16_764.jpg_original.png "Original image from track 2"
[activation_track2]: ./writeup_images/features.png "Activations from image from track 2"
[model]: ./model.png "Model Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The simulator has to be running with the lowest graphics quality settings (Fastest), at a resolution of 640x480. These are the settings under which data was collected and the driving performance was tested.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. You'll find a configuration section on top of the file. Here you can choose which datasets to run through the pipeline, with which hyper-parameters and whether the code will run on a local machine or on floydhub.

```python
# configuration
USE_FLOYD = True
DO_TRAIN = True
SIDE_IMAGE_STEERING_BIAS = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
PREDICT_IMAGES = ["sample_data/center_2016_12_01_13_31_13_177.jpg", "t2_forward_data/center_2017_03_22_17_03_16_764.jpg"]

if USE_FLOYD:
  # running on floydhub
  DATA_FOLDER = "/input/"
  DATASETS = ["t1_reverse_data.csv", "t1_udacity_data.csv", "t1_open_curve.csv"]
  EPOCHS = 7
  OUTPUT_FOLDER = "/output/"
else:
  # running on local machine
  DATA_FOLDER = "../CarND-Behavioral-Cloning-P3-data/multiple_data/"
  DATASETS = ["sample_data.csv"]
  EPOCHS = 2
  OUTPUT_FOLDER = ""
```

Augmentation by flipping can be activated/deactivated in the following function:
```python
def readImagesAndMeasurements(samples, augment=True):
```

The script can also be used to a load previous weights from model.h5 and visualize the activations of hidden layers in the network. By default, this is done for the first few convolutional layers and the resulting feature maps are saved in the folder feature_maps. A sample folder is commited to this github repository, but it will be overwritten on the first run of model.py. The PREDICT_IMAGES setting allows to choose images that are used to get the activations. A sample can be seen here:

!["Original image from track 2"][sample_track2]
*Original image from track 2*

!["Current Model Training History"][activation_track2]
*Activations from image from track 2*


#### 4. Training Data Format

The format I'm storing training data in had to be changed, mainly to integrate better with floydhub. I'm still using the same CSV file and the same image format, but these files are organized in a different way:

* Images of a run are kept in zip files, named \<data-id\>.zip
* The links to images in CSV files now follow a strict rule, namely: \<data-id\>/my_img_filename.zip

This way, I can mix various datasets by just setting the list of data-ids in my configuration section in the model.py file. Using zip files allows me to upload larger sets of images to floydhub. My CSV files as well as the zipped images are contained in the <DATA_FOLDER>, which can also be set in model.py


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 1x1, 3x3 and 5x5 filter sizes and depths between 3 and 64 (model.py, function getNVIDIAModel()): It's the NVIDIA model taken from the project introduction video. I experimented with the proposed models (a basic one to see whether training works at all and LeNet) and ended up using the NVIDIA model, as the increase in performance compared to LeNet was clearly visible.

* The model includes RELU layers to introduce nonlinearity after every single convolutional layer. Example:
```python
model.add(Convolution2D(64, 3, 3, activation="relu"))
```

* The model is normalized in the model using a Keras lambda layer. Here is the layer implementation I used (as explained in the intro videos):
```python
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
```

![Model Architecture][model]
*Model Architecture*

I used dropout in the flat layers to combat overfitting. I experimented with L2 regularization but there were no positive effects from that (as described later in this document). Of course, as described in the intro video, I crop the images using the builtin layer from keras. 

An example of dropout and cropping:

```python
model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
```

Just like in the previous project, I use additional conv layers in the network that are designed to choose the right color space, instead of choosing the color space manually. It's interesting to see which space the network finds most useful. (This can be seen in the folder ./feature_maps)

The following layers are designed to look for the right color space (a 1x1 filter with depth 10, followed by a 1x1 filter with depth 3):

```python
model.add(Convolution2D(10, 1, 1, activation="relu"))
model.add(Convolution2D(3 , 1, 1, activation="relu"))
```


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. These layers are positioned between the flat, fully connected layers of the network. I did not use dropout layers between convolution layers. The steering signals seemed weaker in these cases and were not strong enough to keep the car on track even in slight curves.

The dataset in use was split to training and validation with a 0.2 ratio. This parameter can be set in the following code line in the configuration section of the model.py file:
```python
VALIDATION_SPLIT = 0.2
```

The actual splitting happens here, using a function from sklearn:
```python
train_samples, validation_samples = train_test_split(samples, test_size=VALIDATION_SPLIT)
```

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The adam optimizer is chosen using the following code:
```python
# choose loss function and optimizer, compile the model
model.compile(loss='mse', optimizer='adam')
```

The final model was trained for 7 epochs with a batch size of 32

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first follow the instructions in the intro videos as precisely as possible to get to a good initial solution and then tweak model and data to get a better driving behaviour and also enable the model to drive on track 2.

So I quickly ended up using the NVIDIA model as a basis and added dropout layers, L2 regularization as well as two consecutive 1x1 filters on top to let the model choose the best color representation.

I applied augmentation by flipping images and measurements:
```python
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement*-1.0)
return augmented_images, augmented_measurements
```

I used all three camera images, with a steering bias of 0.2 initially, later experimented with 0.4. This parameter can be set in the following line in the configuration section:
```python
SIDE_IMAGE_STEERING_BIAS = 0.4
```

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Unfortunately, the MSE loss function never seemed to be a good indicator for how well the model would perform. Sample history graphs can be seen here:

!["Current Model Training History"][current_model_history]
*Current Model Training History*

I recorded two reverse laps on track 1 and trained the model on a combination of the Udacity data and my self-recorded laps. The results looked good but the car kept leaving the track in the half-open curve after the bridge.

To combat this, I recorded multiple runs of forward and backward driving data from that sector. This already resulted in a good overall performance on the whole track. The car would not stick 100% to the track center, but would never leave the track, even at a speed of 20mph

On to track 2, I recorded two laps of forward driving (center driving). When training the model on track 2 only, the results are very promising. The car stays on the center of the road and can drive through the whole track without any problems. However, when using this model or when using models that were trained on track 1 and track 2 together, the performance on track 1 decreases dramatically. The model seems to concentrate too much on the center line, which is missing on track 1. 

To combat the overfitting, I added L2 regularization in every layer with a beta value of 0.01. An example can be seen here:
```python
model.add(Dense(100, W_regularizer=l2(0.01)))
```

This resulted in beautiful training history graphs (example below), but unfortunately the driving performance was really bad. The model had a very strong bias on straight driving and would leave the track in the first curve. So, L2 regularization was removed.

!["Training history with L2 regularization"][history_with_l2]
*Training history with L2 regularization*

The results can be seen in the [Autonomous Driving Video](./video.mp4)

#### 2. Final Model Architecture

The final model architecture is described in this document under "Model Architecture and Training Strategy".

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two reverse laps on track one using center lane driving. Here is an example image of center lane driving.

!["Center Driving, center camera"][center_driving_center]

I used the data provided by Udacity in combination with my own data.

I never had to record recovery driving, as I was using imagery from all three cameras with steering offsets. This yielded stable recoveries from the sides. The same image of center driving, seen from all three cameras:

!["Center Driving, left camera"][center_driving_left]
*Center Driving, left camera*

!["Center Driving, center camera"][center_driving_center]
*Center Driving, center camera*

!["Center Driving, right camera"][center_driving_right]
*Center Driving, right camera*

For track 2, I recorded two laps of forward driving on the center line. 

To augment the data set, I flipped images and angles and added these new data points. This was necessary because especially track 1 has a strong bias on left turns, but the model should generalize over left and right turns. Images and measurements are flipped on-the-fly, the code is shown under "Model Architecture and Training Strategy, section 1".

Another shortcoming of the dataset is that steering information (especially sharp turns) are relatively sparse. I therefore added an oversampling function, which will add csv lines to the dataset multiple times, depending on the absolute value of the steering measurement. I experimented with ln(abs(measurement)) and the measurement as is as multipliers:
```python
def oversample(samples):
[...]
```

applied here:
```python
# read csv data and oversample to account for imbalanced dataset
samples = oversample(getCSVLinesFromDatasets(DATA_FOLDER, DATASETS))
```

I experimented with adding / removing oversampling, however the effects were not obvious.

### Problems / Open Points
* It was not possible to reproduce training results, as setting the random seed was never really possible. It seems that keras/tensorflow have random generators that don't rely on the random seed that you can set from outside.
* An open point is track 2. Although I succeeded in creating models that could cope with track 2, I never got one that could handle both tracks. I suspected that the model was overfitting to one of the tracks and tried to reduce that but the results varied too much and there wasn't an obvious convergence towards a solution.
* MSE loss is not a good indicator for how well the model drives. The training and validation losses are very misleading, maybe due to the limited amount of data or due to the imbalance in the data. The only way to tell whether a model worked was to start up the simulator.

