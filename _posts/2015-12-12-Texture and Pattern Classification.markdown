---
layout: post
title:  "Texture and Materials Classification using Deep Neural Networks"
date:   2016-01-01 21:41:18 +1100
categories: Deep Learning
---
In this post, I will share my experience as a Deep Learning researcher beginner while I am working as a Research Assistant at NICTA (National ICT Australia) under Dr. Cong Phuoc Hyunh to classify Textures and Materials from images.

## Data
As machine learning is a data driven approach, we hence need data. There are quite a few materials and textures dataset and we use namely <a href="http://www.robots.ox.ac.uk/~vgg/data/dtd/">Descibable Textures Dataset (DTD)</a>, <a href='http://people.csail.mit.edu/celiu/CVPR2010/FMD/'>Flickr Materials Database (FMD)</a> and <a href='http://opensurfaces.cs.cornell.edu/publications/minc/'>Materials in Context Database (MINC)</a>. DTD, FMD and MINC have 47, 10 and 23 classes respectively. DTD has 120x47=5640 images, FMD has 100x10=1000 images and MINC has 2996674 patches. DTD and FMD has equal number of images per class while MINC all classes have different number of images. For instance, Wood has over half a million patches, while Wallpaper has just over 14000 images.

## Hardware
It is often said more data beats better algorithms. Hence, deep learning is able to exploit lots of data to reveal patterns in the data. Hence to take advantage of this massive amount of data, we need parallel GPU computing. In our case, we used a Tesla K40 and GTX 980 GPU which have 12 GB and 4 GB VRAM respectively.
