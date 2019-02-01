# Behavioral Cloning Project

The steps executed in this project are the following:

  *  Use the simulator to collect data of good driving behavior

  * Build, a convolution neural network in Keras that predicts steering angles from images

  * Train and validate the model with a training and validation set

  * Test that the model successfully drives around the track without leaving the road

## Model Architecture and Training Strategy

  I experimented with 30+ variations of data, model architecture and training parameters.  I ran each combination multiple types as a   network may end up poorly trained due to the random weights initialized or due to the way in which the data was shuffled randomly.  Once a good data set was collected, multiple networks were able to drive one lap around the track without going outside the track – but some architecture options gave better and more consistent results.

### 1. DATA
My data collection strategy involved the following: 
- Running the simulator in training mode on my laptop with 400x300 screen resolution and ‘Fastest’ graphics quality. 
- Recording data in short clips and pickling the data.  Multiple pickle files were created so that the data could be chosen and assembled in a modular manner and minimizing the amount of large file data uploads to the AWS GPU 

The following are the observations: 

* Normal anticlockwise driving data: Was attempted at 30 mph and 10mph - the latter gave better results due to smoother driving. 
      
* Clockwise driving data : 1 round of driving clockwise around the track was recorded

* Recovery driving : Recovery driving data was recorded in specific sections of the track. After a few autonomous driving videos were seen, more recovery data was recorded in areas of sharp turns, unusual mud patches, plants etc. where the car veered off the track. It was important that the recovery image rows were proportional to the normal driving rows. In the best model, recorded recovery driving rows were quadrupled to be impactful in the learned model. 

* Flipping Images: Flipping images also resulted in a successful driving lap but the best model did not use flipping of images – maybe because the clockwise driving data was adequate

* Cropping Images: Cropping the top 50 pixels and bottom 20 pixels gave good results. Cropping the top 60 pixels was not more effective than cropping 50 pixels. 


### 2. MODEL ARCHITECTURE

I experimented with various convolutional layers and filters, use of dropout. 

* Number of convolutional layers: 2 layers with max pooling gave good results. Did not experiment beyond 3 layers as 2 layers performed better than 3 layers. 

* Filter size: I tried 3x3, 5x5, 7x7, 10x10 filters.  5x5 gave the best results. Rectangular filters were tried as lane lines are vertical – gave reasonable results but not an improvement over square filters. 

* Number of filters: I tried using combinations of 4, 8, 12, 16, 24 and 32 filters. Using 8 filters in the first layer and 16 in the second layer gave good results. 

* Dropout: To reduce overfitting, dropout of 0.1, 0.2, 0.5 were tried.  0.1 and 0.2 gave successful laps but the model with the smoothest drive did not use dropout. 

### 3. TRAINING PARAMETERS

As this is a regression model, MSE was used as the error metric. The model used an Adam optimizer, so the learning rate was not tuned manually.  The data was shuffled and 20% of the data was used for validation.  It was found that training of 3-5 epochs gave the best results.  Even using dropout, overfitting was observed beyond 5 epochs. 

Another observation is that validation MSE is not a good indicator of whether a lap will successfully complete or not.  This is because of multiple reasons: 



- The training data does not always use the best turning angle at each step. My driving using a keyboard involved straight driving primarily with turns as required – e.g. 0, 0, -1.5 when in reality -0.5, -0.5, -0.5 would have been better driving. Hence, the training data is flawed at an individual row level though good when averaged. 

- The car may drive close to the training driving for most of the track but may fail badly at one specific place.  Failing in a lap is impacted more by a ‘local MSE’ rather than an ‘overall MSE’.  Validation MSE as an error metric does not measure consistency of MSE across the track. 

The submitted model uses 2 convolutional layers: 

- 8  5x5 filters with RELU activation followed by max pooling

- 16 5x5 filters with RELU activation followed by max pooling

- No dropout 

- A 128 node fully connected layer followed by a single node output layer

![GIF](./examples/CARND_BEHAVIORCLONING_gif.gif)



# Original Readme

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.



We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to 

train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

