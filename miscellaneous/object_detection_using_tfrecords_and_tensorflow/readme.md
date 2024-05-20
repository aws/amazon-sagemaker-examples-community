# Building an Object Detection Model with TensorFlow and TFRecords Using Amazon SageMaker

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your SageMaker service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/). You will need a minimum limit of 1 ```ml.p3.2xlarge``` instance types. Keep in mind that the service limit is specific to each AWS region. We recommend using ```us-west-2``` region for this tutorial.

3. Create an [Amazon SageMaker domain](https://docs.aws.amazon.com/sagemaker/latest/dg/sm-domain.html) in the AWS region where you would like to execute this tutorial. 

## Training the Object Detection Model

In this tutorial, our focus on using [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) for [TensorFlow](https://github.com/tensorflow/tensorflow) using [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

Specifically, we will discuss TFRecord file creation and data preparation using the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

This tutorial consists of two notebooks:

1. Object_detection_preprocessing - you will preprocess the data, then create a json file from the cleaned data. By the end of part 1, you will have a complete data set that contains all features used on Object selection to be ingested by a data loader in *[TensorFlow](https://github.com/tensorflow/tensorflow)* using 'TFRecords'.

2. Object_detection_train_eval - you will use the data set built from part 1 to create a data loader for Tensorflow using *[Keras CV](https://github.com/keras-team/keras-cv)*, train the model and then test the model predictability with the test data. 
