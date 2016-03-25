---
layout: post
title:  "Neural Networks using Tensorflow (Part 1: Logistic Classifier)"
date:   2016-03-19 00:00:00
categories: Deep Learning
---

## Introduction
These days we hear about Deep Learning everywhere, from <a href="https://hbr.org/2016/03/alphago-and-the-limits-of-machine-intuition">AlphaGo </a>defeating 9-dan ranked Lee Sedol in a game of Go 4 out of 5 times, to Google revealing <a href="http://www.theverge.com/2016/2/25/11112594/google-new-deep-learning-image-location-planet">PlaNet</a> that can figure out a photo's location simply by looking at it. 
The field of Deep Learning, which first started in the 80s and continued into the 90s is now gaining popularity because of the availability of powerful GPUs and huge amounts of Data. Some of the very popular Deep Learning frameworks are <a href="http://caffe.berkeleyvision.org/">Caffe</a>, <a href="https://www.tensorflow.org/">Tensorflow</a>, <a href="http://torch.ch/">Torch</a> and <a href="http://www.cntk.ai/">CNTK</a>. Each framework has its own advantages and disadvantages.

The essential core of Deep Learning is a Neural Network that learns multiple levels of hierarchical features. Before we dive into neural networks, we will look at a simple logistic classifier. This will be the focus of the current post.


## Classification
Classification, one of the building blocks of Machine Learning, is the task of taking an input and assigning it a label. For example, we can take several images of cats and dogs and train a model using those images. ![Train-images]({{site.url}}/assets/Train-images.png) Finally, when we input a new image into the model, it should tell us that the label/class of the image is Cat. ![Test-Cat]({{site.url}}/assets/Test-cat.jpg).

### Logistic Classifier
Logistic Classifier is one of the simplest classifiers. It is a linear classifier $$WX + b = Y$$ where $$X$$ is the input (e.g. pixels in an image) and $$Y$$ are the predictions. During the training phase, we try to learn the weight matrix $$W$$ and bias $$b$$ from our training data. We want the learnt $$W$$ and bias $$b$$ to be good at making predictions.

As we are multiplying matrices, we expect $$Y$$ to be a column matrix of real numbers. In other words, if we have 3 classes we will get a $$Y$$ with 3 rows where each row represents a score for a class. We also know each input $$X$$ can have 1 class only. Hence, to perform classification, we have to convert these scores into probabilities. We want the probability of the correct class to be close to 1 and the probability of the incorrect class be close to 0. All these scores can be converted to probabilities using a SoftMax function. 
The beauty of this function is that it can convert any kind of scores to proper probabilities. These scores are also called logits in the context of logistic regression.

The softmax function is defined as:
$$\sigma(y_i)=\frac{e^{y_i}}{\sum_{j}{e^{y_j}}}$$ where $$j$$ is the number of classes.


$$WX+b= \begin{bmatrix} 4\\ 1\\ 0.1\\ \end{bmatrix} \xrightarrow{softmax} \begin{bmatrix} 0.7\\ 0.2\\ 0.1\\ \end{bmatrix}$$

#### One Hot Encoding
The question is now: How do we represent the label of an image mathematically? In other words, how do we tell the computer that a particular matrix $$X$$ represents a Cat, Dog or Zebra? This is done using a column vector and it is as long as the number of classes. Each row in the column vector represents a particular class. For the correct class, the vector has 1 in the row represented for that class and 0 elsewhere. For example, if we have 3 classes, Cat, Dog and Zebra, the one-hot encoded vectors then will be:
 $$\begin{bmatrix} 1\\ 0\\ 0\\ \end{bmatrix}, \begin{bmatrix} 0\\ 1\\ 0\\ \end{bmatrix}, \begin{bmatrix} 0\\ 0\\ 1\\ \end{bmatrix}$$ respectively.
We will use these one-hot encoded vectors as labels.

#### Cross Entropy
So now we have 2 sets of numbers: the first is the output of the classifier (i.e. $$\sigma(WX+b)$$) and the one hot-encoded vectors that correspond to our labels. To measure how well our classifier is doing, we have to measure the distance between the 2 sets of numbers. We can do that by using **cross entropy**. Let's represent the output of the classifier as $$S$$ and labels as $$L$$. 
The distance is represented as:
$$D(S,L) = -\sum_{i}L_i.log(S_i)$$. We do not want the one-hot encoded labels to be under the $$log$$ as they contain zeros.


To summarise:
<ol>
<li> We have an input X </li>
<li> We turn X into logits using a linear model</li>
<li> We then feed the logits into a softmax function and turn them into probabilities</li>
<li> Finally, we compare the probabilities with the one-hot encoded labels using the cross entropy function.</li>
$$X \xrightarrow{WX+b} \begin{bmatrix} 4\\ 1\\ 0.1\\ \end{bmatrix} \xrightarrow{softmax} \begin{bmatrix} 0.7\\ 0.2\\ 0.1\\ \end{bmatrix} \xrightarrow[Cross-Entropy]{D(S,L)} \begin{bmatrix} 1\\ 0\\ 0\\ \end{bmatrix}$$
</ol>

The above 4 settings are called **Multinomial Logistic Classification**.

#### Learning $$W$$ and $$b$$
The next step is to learn $$W$$ and $$b$$ from our training data. In other words, we need a $$W$$ and $$b$$ that has low distance for the correct class but a high distance from an incorrect class. We can measure the distance averaged over our entire training set and we call it the **training loss**. In other words, loss = average cross entropy and it is a function of weights $$W$$ and biases $$b$$.
$$L = \frac{1}{n}\sum_{i}D(S(WX_i+b), L_i)$$. We want this loss to be as small as possible and hence during training we want to minimize this loss function $$(L)$$. Hence our problem has become a numerical optimisation problem. In other words, we try to find the weights that cause the loss to be smallest. 

There are several ways to achieve this minimisation. The simplest way is **gradient descent**. In gradient descent, we take the derivative of the loss with respect to the parameters and follow the derivative by taking a step backwards and repeating until we reach the bottom. In the diagram below, the loss is a function of 1 parameter $$W$$ only. In real world problems, it will have millions of parameters.
![Gradient-Descent]({{site.url}}/assets/Gradient-Descent.png)

#### Preprocessing Input and Initialising Weights
Before we feed our training data into the classifier, if possible we would like the data to have zero mean and unit variance. We do this to help the optimiser find the optimal solution quickly. In other words, in a badly conditioned problem, the optimiser will have to do lots of searching to find the optimal solution. To achieve this in the case of images, we subtract each pixel value by 128 and then divide it by 128.

To intialise the weights, we simply randomly draw numbers from a gaussian distribution with zero mean and a small $$\sigma$$. We do this because, if we have large $$\sigma$$, the model will be very confident in its prediction. We don't want that initially. Instead we want our model to gain confidence as the training progresses, hence we choose a small value for $$\sigma$$.

Finally, the optimisation looks like:

$$W \leftarrow W-\alpha\Delta_w L$$

$$b \leftarrow b-\alpha\Delta_b L$$

We repeat the above 2 steps until we reach the minimum loss.

## Code
In this part, we will look into how we apply the same in [scikit-learn](http://scikit-learn.org/)  and will use the data [noMNIST]('http://yaroslavvb.com/upload/notMNIST/') dataset. This dataset contains characters in various fonts with $$28\times28$$ features.

First we import the libraries, 

{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle
{% endhighlight %}

Next we download and extract the dataset,

{% highlight python %}
url = 'http://yaroslavvb.com/upload/notMNIST/'

def download(filename, expected_bytes):
  """Download a file if not present"""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print 'Found and verified', filename
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = download('notMNIST_large.tar.gz', 247336696)
test_filename = download('notMNIST_small.tar.gz', 8458043)


num_classes = 10

def extract(filename):
  tar = tarfile.open(filename)
  tar.extractall()
  tar.close()
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_folders, len(data_folders)))
  print data_folders
  return data_folders
  
train_folders = extract(train_filename)
test_folders = extract(test_filename)
{% endhighlight %}

Now, we load the data into a 3D array (index, height, width) and create a training set and a test set.

{% highlight python %}
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load(data_folders, min_num_images, max_num_images):
  dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
  labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
  label_index = 0
  image_index = 0
  for folder in data_folders:
    print folder
    for image in os.listdir(folder):
      if image_index >= max_num_images:
        raise Exception('More images than expected: %d >= %d' % (
          num_images, max_num_images))
      image_file = os.path.join(folder, image)
      try:
        image_data = (ndimage.imread(image_file).astype(float) -
                      pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_size, image_size):
          raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[image_index, :, :] = image_data
        labels[image_index] = label_index
        image_index += 1
      except IOError as e:
        print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
    label_index += 1
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_images]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' % (
        num_images, min_num_images))
  print 'Full dataset tensor:', dataset.shape
  print 'Mean:', np.mean(dataset)
  print 'Standard deviation:', np.std(dataset)
  print 'Labels:', labels.shape
  return dataset, labels
train_dataset, train_labels = load(train_folders, 450000, 550000)
test_dataset, test_labels = load(test_folders, 18000, 20000)
{% endhighlight %}

Next we display the images,
{% highlight python %}
a = Image('notMNIST_large/A/aGFua28udHRm.png')
b = Image('notMNIST_large/B/aGFua28udHRm.png')
c = Image('notMNIST_large/C/aGFua28udHRm.png')
d = Image('notMNIST_large/D/aGFua28udHRm.png')
e = Image('notMNIST_large/E/aGFua28udHRm.png')
f = Image('notMNIST_large/F/aGFua28udHRm.png')
g = Image('notMNIST_large/G/aGFua28udHRm.png')
h = Image('notMNIST_large/H/aGFua28udHRm.png')
i = Image('notMNIST_large/I/aGFua28udHRm.png')
j = Image('notMNIST_large/J/aGFua28udHRm.png')

display(a,b,c,d,e,f,g,h,i,j)
{% endhighlight %}

![Characters]({{site.url}}/assets/Characters.png)

It is very important to randomise the dataset so that our model during training does not get biased towards a particular label. We do this as:
{% highlight python %}
np.random.seed(133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
{% endhighlight %}

Now we separate our training set into 2 set: `train set`, `validation set`. We use the validation set for hyper-parameter tuning.

{% highlight python %}
train_size = 200000
valid_size = 10000

valid_dataset = train_dataset[:valid_size,:,:]
valid_labels = train_labels[:valid_size]
train_dataset = train_dataset[valid_size:valid_size+train_size,:,:]
train_labels = train_labels[valid_size:valid_size+train_size]
print 'Training', train_dataset.shape, train_labels.shape
print 'Validation', valid_dataset.shape, valid_labels.shape
{% endhighlight %}

We save this model on a disk for re-use later.

{% highlight python %}
pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print 'Unable to save data to', pickle_file, ':', e
  raise
{% endhighlight %}

Now comes the training phase. We are going to use [scikit-learn](http://scikit-learn.org/) to train our model.
`scikit-learn` takes in data as a `2D` array only (i.e. `(n_samples, n_features)`). On the other hand, our dataset is a `3D` array `(n_samples, height, width)`. Hence, we are going to flatten each `2D` image into a `1D` array (i.e. a `28 x 28` values in an image is now a `1D` array of `784` numbers/features). We will do this in code as:

{% highlight python %}
train_dataset = [arr.flatten() for arr in train_dataset]
test_dataset  = [arr.flatten() for arr in test_dataset]
{% endhighlight %}

Finally, we do the training using 50, 1000, 5000 training samples and report the accuracy in the test set.

{% highlight python %}
for n_samples in [50, 1000, 5000]
    # Training
    clf = LogisticRegression()
    clf.fit(train_dataset[:n_samples], train_labels[:n_samples])
    # Testing
    print(clf.score(test_dataset, test_labels))
{% endhighlight %}

Using 50, 1000 and 5000 training we get a test accuracy of 63.8%, 74.9% and 84.7% respectively. This shows that more data we have, the better is the generalization.

In the Part 2 of this series, we will train a model on the same data set using a Feed Forward Neural Network using Tensorflow and we will see an increase in the test accuracy.









