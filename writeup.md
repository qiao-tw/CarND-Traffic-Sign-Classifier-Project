# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./histgram.png "samples in each class"
[image10]: ./color_vs_gray.png "samples in each class"
[image11]: ./download/web/12.jpg "sample from web"
[image12]: ./download/web/13.jpg "sample from web"
[image13]: ./download/web/14.jpg "sample from web"
[image14]: ./download/web/17.jpg "sample from web"
[image15]: ./download/web/22.jpg "sample from web"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because using RGB color images can't get an good result - accuracy always less than 0.9.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image10]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:-----------------:|:---------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 5x5x16 				|
| Flatten           | input  5x5x16, output = 400         |
| Fully connected		| input = 400, output = 120 	  			|
| RELU					    |             												|
| Fully connected		| input = 120, output = 84  	  			|
| RELU					    |             												|
| Fully connected		| input = 84, output = 43   	  			|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used mini-batch (BATCH_SIZE = 128) SGD approach to train the network.

I used Adam method to optimize the loss function (cross entropy).

The number of epochs is 200, I set a break point when validation accuracy reach 0.93 in order to reduce execution time.

Learning rate is 0.001.

Within each epoch, the accuracy in validation are printed out. The accuracy is continually increasing in most of the epochs. At the last epoch(EPOCH 30), validation accuracy reach around 0.932, which is a satisfactory result.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.932
* test set accuracy of 0.906

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15]

Image(1) and Image(2) have noisy background; Image(3) is a little tilt (since I didn't apply any argumentation for training, it's no surprised if this image fails in prediction). Image(4), Image(5) looks OK for prediction.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:-----------------:|:---------------------------------:|
| Priority Road     | Priority Road   									|
| Yield   					| Yield	        										|
| Stop    					| Stop	        										|
| No Entry      		| No Entry	        				 				|
| Bumpy Road	  		| Bumpy Road          							|

The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the 1st image, the model is very sure that this is a "Priority Road" sign (probability reach ~ 1.0), and the image does contain a "Priority Road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| ~ 1.0           			| Priority Road   									|
| 6.8e-24       				| Yield         										|
| 9.4e-26       				| Speed Limit (50km/h) 							|
| 9.4e-28       				| Ahead only 					              |
| 1.3e-30       				| No Passing 										    |

For the 2nd image, the model is very sure that this is a "Yield" sign (probability of ~ 1.0), and the image does contain a "Yield" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| ~ 1.0           			| Yield   					        				|
| 7.0e-27       				| Dangerous Curve to the Right 			|
| 9.4e-29       				| Keep Right 			 							    |
| 7.0e-30       				| Turn Right Ahead			          	|
| 3.1e-31       				| No Vehicles     								  |

For the 3rd image, the model is very sure that this is a "Stop" sign (probability of ~ 1.0), and the image does contain a "Stop" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| ~ 1.0           			| Stop   									|
| 2.2e-08       				| Speed Limit (30km/h)		|
| 7.3e-09       				| Speed Limit (50km/h)		|
| 2.4e-12       				| Bicycles Crossing  			|
| 1.7e-12       				| Roundabout Mandatory		|

For the 4th image, the model is very sure that this is a "No Entry" sign (probability of ~ 1.0), and the image does contain a "No Entry" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| ~ 1.0           			| No Entry   							|
| 9,8e-35       				| Turn Left Ahead					|
| 4.8e-36       				| Yield 									|
| 2.6e-36       				| Bicycles Crossing				|
| 5.3e-37       				| Traffic Signals					|

For the 5th image, the model is vert sure that this is a "Bumpy Road" sign (probability of ~ 1.0), and the image does contain a "Bumpy Road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| ~ 1.0           			| Bumpy Road   									|
| 1.3e-28       				| Traffic Signals								|
| 2.2e-34       				| Road Work 										|
| 3.9e-35       				| Bicycles Crossing 						|
| 1.7e-37       				| Children Crossing 						|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
