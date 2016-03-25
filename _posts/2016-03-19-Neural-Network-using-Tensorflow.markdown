---
layout: post
title:  "Neural Networks using Tensorflow (Part 1: Logistic Classifier)"
date:   2016-03-19 00:00:00
categories: Deep Learning
---

## Introduction
These days we hear about Deep Learning everywhere from <a href="https://hbr.org/2016/03/alphago-and-the-limits-of-machine-intuition">AlphaGo </a>defeating 9-dan ranked Lee Sedol in game of Go 4 out of 5 times to Google revealing <a href="http://www.theverge.com/2016/2/25/11112594/google-new-deep-learning-image-location-planet">PlaNet</a> which can figure out the location of a photo just by looking at it. 
The field of Deep Learning which first started in 80s continuing to 90s is now getting popularity because of the availability of powerful GPUs and huge amounts of Data. Some of the very popular Deep Learning frameworks are <a href="http://caffe.berkeleyvision.org/">Caffe</a>, <a href="https://www.tensorflow.org/">Tensorflow</a>, <a href="http://torch.ch/">Torch</a> and <a href="http://www.cntk.ai/">CNTK</a>. Each framework have their own advantages and disadvantages.

The very core of Deep Learning is a Neural Network which learns multiple levels of hierarchical features. In this post, we will have a look how to train a simple feed forward neural network using Tensorflow.


## Classification
Classification one of the building blocks of Machine Learning is the task of taking an input and assigning it a label. For e.g. We take several images of cats and dogs and train a model using those images. ![Train-images]({{site.url}}/assets/Train-images.png) Finally, when we input a new image to the model, it should give us the label/class of the image is Cat. ![Test-Cat]({{site.url}}/assets/Test-cat.jpg).

### Logistic Classifier
Logistic Classifier is one of the simplest classifier. It's a linear classifier $$WX + b = Y$$ where $$X$$ is the input (e.g. pixels in an image) and $$Y$$ are the predictions. During the training phase, we try to learn the weight matrix $$W$$ and bias $$b$$ from our training data. We want the learnt $$W$$ and bias $$b$$ are good at making predictions.

As we are multiplying matrices, we expect $$Y$$ to be a column matrix of real numbers. In other words, if we have 3 classes we will get a $$Y$$ with 3 rows where each row represent a score for a class. We also know each input $$X$$ can have 1 class only. Hence to perform classification we have to turn these scores into probabilities. We want the probability of the correct class should be close to 1 and the probability of the incorrect class be close to 0. All these scores can be converted to probabilities using a SoftMax function. 
The beauty of this function is that it can take any kind of score to proper probabilities. These scores are also called logits in the context of logistic regression.

The softmax function is defined as:
$$\sigma(y_i)=\frac{e^{y_i}}{\sum_{j}{e^{y_j}}}$$ where $$j$$ is the number of classes.


$$WX+b= \begin{bmatrix} 4\\ 1\\ 0.1\\ \end{bmatrix} \xrightarrow{softmax} \begin{bmatrix} 0.7\\ 0.2\\ 0.1\\ \end{bmatrix}$$

#### One Hot Encoding
The question is now: How do we represent the label of an image mathematically? In other words, how do we tell the computer a particular matrix $$X$$ represents a Cat or Dog or Zebra? This is doing using a column vector and its as long as the number of classes. Each row in the column vector represents a particular class. For the correct class, the vector has 1 in the row represented for that class and 0 elsewhere. For e.g. if we have 3 classes Cat, Dog and Zebra, the one-hot encoded vectors then will be:
 $$\begin{bmatrix} 1\\ 0\\ 0\\ \end{bmatrix}, \begin{bmatrix} 0\\ 1\\ 0\\ \end{bmatrix}, \begin{bmatrix} 0\\ 0\\ 1\\ \end{bmatrix}$$ respectively.
We will use these one-hot encoded vectors as labels.

#### Cross Entropy
So now we have 2 sets of numbers: the first the output of the classifier i.e. $$\sigma(WX+b)$$ and the one hot-encoded vectors that correspond to our labels. To measure how well our classifier is doing, we have to measure the distance between the 2 sets of numbers. We can do that by using **cross entropy**. Let's represent the output of the classifier is $$S$$ and labels as $$L$$. 
The distance is represented as:
$$D(S,L) = -\sum_{i}L_i.log(S_i)$$. We do not want the one-hot encoded labels under the $$log$$ as they contain zeros.


To summarise:
<ol>
<li> We have an input X </li>
<li> We turn X into logits using a linear model</li>
<li> We then feed the logits into a softmax function and turn them into probabilities</li>
<li> Finally we compare the probabilities with the one-hot encoded labels using the cross entropy function.</li>
$$X \xrightarrow{WX+b} \begin{bmatrix} 4\\ 1\\ 0.1\\ \end{bmatrix} \xrightarrow{softmax} \begin{bmatrix} 0.7\\ 0.2\\ 0.1\\ \end{bmatrix} \xrightarrow[Cross-Entropy]{D(S,L)} \begin{bmatrix} 1\\ 0\\ 0\\ \end{bmatrix}$$
</ol>

The above 4 settings is called **Multinomial Logistic Classification**.

#### Learning $$W$$ and $$b$$
The next step is to learn $$W$$ and $$b$$ from our training data. In other words, we need a $$W$$ and $$b$$ that has low distance for the correct class but have a high distance from an incorrect class. We can measure the distance averaged over for all our entire training set and we call it the **training loss**. In other words, loss = average cross entropy and it is a function of weights $$W$$ and biases $$b$$.
$$L = \frac{1}{n}\sum_{i}D(S(WX_i+b), L_i)$$. We want this loss to be as small as possible and hence during training we want to minimize this loss function $$(L)$$. Hence our problem has become a numerical optimisation problem. In other words, we try to find the weights that cause the loss to be smallest. 

There are several ways to achieve this minimisation. The simplest way is **gradient descent**. In gradient descent, we take the derivative of the loss with respect to the parameters and follow the derivative by taking a step backwards and repeat till we reach the bottom. In the diagram below, the loss is a function of 1 parameter $$W$$ only. In real world problems, it will have a millions of parameters.
![Gradient-Descent]({{site.url}}/assets/Gradient-Descent.png)

#### Preprocessing Input and Initialising Weights
Before we feed our training data to the classifier, if possible we would like the data to have zero mean and unit variance. We do this to help the optimiser to find the optimal solution quickly. In other words, in a badly conditioned problem the optimiser will have to do lots of searching to find the optimal solution. To achieve this in case of images, we subtract each pixel value by 128 and then divide it by 128.

To intialise the weights, we simply randomly draw numbers from a gaussian distribution with zero mean and a small $$\sigma$$. We do this because, if we have large $$\sigma$$, the model will be very confident in its prediction. We don't want that initially, instead we want our model to gain confidence as the training progresses, hence we choose a small value for $$\sigma$$.

Finally, the optimisation looks like:

$$W \leftarrow W-\alpha\Delta_w L$$

$$b \leftarrow b-\alpha\Delta_b L$$

We repeat the above 2 steps until we reach the minimum loss.







