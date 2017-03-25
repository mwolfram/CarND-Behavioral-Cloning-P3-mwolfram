# CarND-Behavioral-Cloning-P3-mwolfram

## notes
* RUNNING SIMULATOR on  640x480, lowest graphics quality, highest performance
* as a starting point, I used the basic model.py from the video
* I used the training data from udacity for the first model
* changed to LeNet
* augmented data by flipping
* used all 3 cameras, steering offset 0.2
* cropping image as described in course
* normalizing images as described in course (lambda layer)
* used NVIDIA network
* added two 1x1 filters to choose color space
* added training data: two reverse laps t1, two forward laps t2, driven with mouse (simulator settings same, 640x480, lowest quality)
* changed the code to run easily on floydhub
* training for 5 epochs, batch size 32
* tried random seed, cannot reproduce the same results, unfortunately. search indicates that this lies deep within keras and tf
* train/validation split ratio 0.2
* optimizer adam, loss function is mean squared error
* python generator is being used
* EXPLAIN: data format expected (directly from zips)
* RESULTS so far: good driving on t1 with datasets t1_udacity, t1_reverse B32 E5 loss: 0.0173 - val_loss: 0.0173
* RESULTS so far: good driving on t2 with datasets t1_udacity, t1_reverse, t2_forward B32 E5 - BUT: terrible on t1!, possible reason: focus on center line, which is missing on t1!
* FURTHER IDEAS: all datasets, but with dropout and L2 reg, layer visualization to know what the network focuses on, graph loss and accuracy
* added dropout, added history visualization, training on floyd. all datasets
* still TO DO: L2 reg, layer visualization.
* FURTHER IDEAS: stronger bias on turning situations (multiply image occurrences by (abs((int)angle)) + 1
* exp15: 20ep 32b dropout 0.5 added (TODO show model, can be printed in kerasss), almost ok on t1 - learning curve will show ideal number of epochs, I'll keep that. problems with side markings that are not those stripes, problem before bridge, centered on bridge, serious issue (leaving road) in last curve (weaker marking, and only on one side. might have to emphasize that training data. curb is touched multiple times. t2: less focus on center line (was a lot more without the dropout and with less epochs!), failes in sharp turn 4x (problem: too much straight driving bias??)
* so TO DO: get the model from the previous run, get optimum number of epochs, then try curve multiplier (attention lot of curves on t2! overfitting!), L2 reg, visualize!
* L2 reg did not seem too helpful: EXP 17: W2_reg and dropout everywhere. nice learning curve but really bad performance, almost no turning on T1, less turning on T2
* added visualization of activations (done with every run, automatically)
* EXP20 to see what's the best combination: t1 all data, 7EP, 32B, L2 and Dropout everywhere
* EXP21 planned: t2 data, same as above
* EXP22 planned: back to the roots, L2 taken out, Dropout drastically reduced, all data (try until t2 is feasible)
* EXP planned: same for t1 data (t1 should then be feasible)


## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
