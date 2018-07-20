# Traffic Light Detection

This a tensorflow implementation of [R-FCN: Object Detection via
Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409.pdf).

## Aim

With the recent launch of the self driving cars and trucks, the field of autonomous navigation has never been more exciting. One of the main tasks that any such vehicle must perform well is the task of following the rules of the road. Identifying the traffic lights in the midst of everything is the one of the most important tasks. 

There are already many real time object detection models for traffic lights which can detect and recognize traffic lights in an image at a frame rate of 7-10 fps. This project aims to build a lightweight version of the traffic light detection model, which is smaller in size but farely accurate and can work faster on embedded devices having relatively less computational power. 

## Strategy

My first idea was to go through all the currently used real time object detection models like SSD, R-FCN and Faster RCNN. In this [paper](https://arxiv.org/abs/1611.10012) they contain a proper analysis of all the architectures. Also it has a detailed analysis of choosing between different feature extractors. So, I chose the R-FCN architecture with Inception-Resnet feature extractor. Though SSD is the fastest of all them all but it is not very accurate for smaller objects in an image. R-FCN is not as fast as SSD but it is farely more accurate than it.

Next major thing was to think of some of ways to reduce the model size even further. In this [paper](https://arxiv.org/abs/1510.00149) authors describe basically three methods: 

1. Network Pruning: In this method a network is fully trained and then any connections with a weight below a certain threshold are removed leaving a sparse network.
2. Trained Quantization and Weight Sharing: Here the weights in a network are clustered together with other weights of similar magnitude, and all these weights are then represented by a single shared value.
3. [Huffman Coding](https://en.wikipedia.org/wiki/Huffman_coding): The general idea is that it uses fewer bits to represent data that appears frequently and more bits to represent data that appears infrequently.

All the methods aim to reduce the size of network quite significantly without reducing accuracy. Besides there are a couple of other simple techniques like dropout which can decrease the size of the network.

On top it we can apply another approach explained in this [paper]() which can reduce the size even further. In this paper, the authors outline 3 main strategies for reducing parameter size while maximizing accuracy.

1. Make the network smaller by replacing 3x3 filters with 1x1 filters
2. Reduce the number of inputs for the remaining 3x3 filters
3. Downsample late in the network so that convolution layers have large activation maps

The only drawback of this approach is that we have to train the whole network from scratch.

Further we can convert the trained model through [tensorflow lite](https://www.tensorflow.org/mobile/tflite/) into a tensorflow lite model file. Then we can use the lite model to classify images.

## Work Done

My first step was to get familiarized with a Deep Learning Framework. I chose tensorflow, as suggested by the mentors. So, 
I did all the tutorials as fast as I could. Then I checked out the [object detection apis](https://github.com/tensorflow/models/tree/master/research/object_detection/meta_architectures) provided by tensorflow dedicated to the construction of meta-achitectures like SSD, R-FCN and Faster RCNN. I also went through tutorials on how to [save and restore models through tensorflow].

I went through a lot of research papers, blogs, tutorials, etc. to come to a plane to implement which I have described above.
Though tensorflow is one of the most efficient framework for Deep Learning, still it can be sometimes very hard to figure out how to work things out. So, for some specific things I chose to keras, a deep learning library which uses tensorflow in backend.  

## Code

This repository contains only the complete R-FCN network, but it is yet to be trained to be fully functional. The most difficult part of the implementation was to figure out:
1. how to restore paramters from a trained model
2. how to use that model to take output from an intermediate layer and then attach to one of the meta-architectures to complete the network
3. how to train only the layers in the meta-architecture and leave the feature extractor unchanged

And, I used the [meta-architecture module](https://github.com/tensorflow/models/tree/master/research/object_detection/meta_architectures) of tensorflow object detection api to construct my network. In order to build the R-FCN network, it needed a couple of input functions and parameters. I wrote all the input functions and feature extractor. The complete code is in `main.py`. 

## Final Result

I was able to figure out what I needed to do but I wasn't able to complete it. I have built the network, but it is yet to be trained before it can be used.
