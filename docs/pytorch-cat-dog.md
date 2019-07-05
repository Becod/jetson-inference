<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="pytorch-transfer-learning.md">Back</a> | <a href="pytorch-cat-dog.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning</sup></s></p>

# Re-training with the Cat/Dog Dataset

The first model that we'll be re-training is a simple model that recognizes two classes:  cat or dog.

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-cat-dog.jpg" width="700">

Provided below is a 750MB dataset that includes 5000 training images, 1000 validation images, and 200 test images, each split evenly between the cat and dog classes.  The set of training images is used for transfer learning, while the validation set is used to evaluate model performance during training, and the test images are to be used by us after training completes.  The network is never directly trained on the validation and test sets, only the training set.

To get started, first make sure that you have [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch), then download the dataset and kick off the training script.

## Downloading the Data

During this tutorial, we'll store the datasets under a common location, like `~/datasets`.  You can store them wherever your wish, just substitute your desired path for `~/datasets` during the steps below.

``` bash
$ mkdir ~/datasets
$ cd ~/datasets
```

Next, download and extract the data:

``` bash
$ wget https://nvidia.box.com/shared/static/o577zd8yp3lmxf5zhm38svrbrv45am3y.gz -O cat_dog.tar.gz
$ tar xvzf cat_dog.tar.gz
```

Mirrors of the dataset are available here:

* <a href="https://drive.google.com/file/d/16E3yFvVS2DouwgIl4TPFJvMlhGpnYWKF/view?usp=sharing">https://drive.google.com/file/d/16E3yFvVS2DouwgIl4TPFJvMlhGpnYWKF/view?usp=sharing</a>
* <a href="https://nvidia.box.com/s/o577zd8yp3lmxf5zhm38svrbrv45am3y">https://nvidia.box.com/s/o577zd8yp3lmxf5zhm38svrbrv45am3y</a>

## Training the Model

The PyTorch training scripts are located in the repo under <a href="https://github.com/dusty-nv/jetson-inference/tree/master/python/training/imagenet">`jetson-inference/python/training/imagenet/`</a>.  These scripts aren't specific to any one dataset, so we'll use the same PyTorch code for each of the example datasets from the tutorial.  By default it's set to train a ResNet-18 model, but you can change that with the `--arch` flag.

To launch the training, run the following commands:

``` bash
$ cd jetson-inference/python/training/imagenet
$ python train.py --model-dir=cat_dog ~/datasets/cat_dog
```

As training begins, you should see text from the console like the following:

``` bash
Use GPU: 0 for training
=> dataset classes:  2 ['cat', 'dog']
=> using pre-trained model 'resnet18'
=> reshaped ResNet fully-connected layer with: Linear(in_features=512, out_features=2, bias=True)
Epoch: [0][  0/625]	Time  0.932 ( 0.932)	Data  0.148 ( 0.148)	Loss 6.8126e-01 (6.8126e-01)	Acc@1  50.00 ( 50.00)	Acc@5 100.00 (100.00)
Epoch: [0][ 10/625]	Time  0.085 ( 0.163)	Data  0.000 ( 0.019)	Loss 2.3263e+01 (2.1190e+01)	Acc@1  25.00 ( 55.68)	Acc@5 100.00 (100.00)
Epoch: [0][ 20/625]	Time  0.079 ( 0.126)	Data  0.000 ( 0.013)	Loss 1.5674e+00 (1.8448e+01)	Acc@1  62.50 ( 52.38)	Acc@5 100.00 (100.00)
Epoch: [0][ 30/625]	Time  0.127 ( 0.114)	Data  0.000 ( 0.011)	Loss 1.7583e+00 (1.5975e+01)	Acc@1  25.00 ( 52.02)	Acc@5 100.00 (100.00)
Epoch: [0][ 40/625]	Time  0.118 ( 0.116)	Data  0.000 ( 0.010)	Loss 5.4494e+00 (1.2934e+01)	Acc@1  50.00 ( 50.30)	Acc@5 100.00 (100.00)
Epoch: [0][ 50/625]	Time  0.080 ( 0.111)	Data  0.000 ( 0.010)	Loss 1.8903e+01 (1.1359e+01)	Acc@1  50.00 ( 48.77)	Acc@5 100.00 (100.00)
Epoch: [0][ 60/625]	Time  0.082 ( 0.106)	Data  0.000 ( 0.009)	Loss 1.0540e+01 (1.0473e+01)	Acc@1  25.00 ( 49.39)	Acc@5 100.00 (100.00)
Epoch: [0][ 70/625]	Time  0.080 ( 0.102)	Data  0.000 ( 0.009)	Loss 5.1142e-01 (1.0354e+01)	Acc@1  75.00 ( 49.65)	Acc@5 100.00 (100.00)
Epoch: [0][ 80/625]	Time  0.076 ( 0.100)	Data  0.000 ( 0.009)	Loss 6.7064e-01 (9.2385e+00)	Acc@1  50.00 ( 49.38)	Acc@5 100.00 (100.00)
Epoch: [0][ 90/625]	Time  0.083 ( 0.098)	Data  0.000 ( 0.008)	Loss 7.3421e+00 (8.4755e+00)	Acc@1  37.50 ( 50.00)	Acc@5 100.00 (100.00)
Epoch: [0][100/625]	Time  0.093 ( 0.097)	Data  0.000 ( 0.008)	Loss 7.4379e-01 (7.8715e+00)	Acc@1  50.00 ( 50.12)	Acc@5 100.00 (100.00)
```

This output corresponds to the following info:

* Epoch:  an epoch is one complete training pass over the data
	* `Epoch: [N]` means you are currently on epoch 0, 1, 2, ect.
	* The default is to run for 35 epochs, you can change this with the `--epochs=N` flag
* `[N/625]` the current image batch from the epoch that you are on
	* Training images are processed in mini-batches to improve performance
	* The default batch size is 8 images, which can be set with the `--batch=N` flag
	* Multiply the numbers in brackets by the batch size (i.e. batch `[100/625]` -> image `[800/5000]`)
* Time:  processing time of the current image batch (in seconds)
* Data:  disk loading time of the current image batch (in seconds)
* Loss:  the accumulated errors that the model made (expected vs. predicated)
* `Acc@1`:  the Top-1 classification accuracy over the batch
	* Top-1 meaning that the model predicted exactly the correct class
* `Acc@5`:  the Top-5 classification accuracy over the batch
	* Top-5 meaning that the correct class was one of the top 5 outputs the model predicted
	* Since this Cat/Dog example only has 2 classes (Cat and Dog), Top-5 is always 100%
	* Other datasets from the tutorial have more than 5 classes, where Top-5 is valid 

To stop training at any time, you can press `Ctrl+C`.  You can also restart the training again later using the `--resume` and `--epoch-start` flags, so you don't need to wait for training to complete before testing out the model.  Run `python train.py --help` for more information about each option that's available for you to use, including other networks that you can try with the `--arch` flag.

### Model Accuracy

On this dataset of 5000 images, training ResNet-18 takes approximately ~7-8 minutes per epoch on Jetson Nano, or around 4 hours to train the model to 35 epochs and 80% accuracy.  Below is a graph for analyzing the progression of training epochs versus model accuracy:

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-cat-dog-training.jpg" width="700">

At around epoch 30, the ResNet-18 model reaches 80% accuracy, and at epoch 65 it converges on 82.5% accuracy.  With additional training time, uou could further improve the accuracy by increasing the size of the dataset (see the [Generating More Data](#generating-more-data) section below) or by trying more complex models.

By default the script is set to run for 35 epochs, but if you don't wish to wait that long to test out your model, you can exit training early and proceed to the next step (optionally re-starting the training again later from where you left off).  You can also download this completed model that was trained for the full 100 epochs from here:

* <a href="https://nvidia.box.com/s/zlvb4y43djygotpjn6azjhwu0r3j0yxc">https://nvidia.box.com/s/zlvb4y43djygotpjn6azjhwu0r3j0yxc</a>

Note that the models are saved under `jetson-inference/python/training/imagenet/cat_dog/`, including the latest checkpoint and the best-performing model.  You can change the directory the models are saved to by changing the `--model-dir` flag.

## Converting the Model to ONNX

To run our trained model with TensorRT for testing and realtime inference, first we need to convert the PyTorch model into ONNX format so that TensorRT can load it.  <a href="https://onnx.ai/">ONNX</a> is an open model format that supports many of the popular ML frameworks, including PyTorch, TensorFlow, TensorRT, and others, so it simplifies transferring models between tools.

PyTorch comes with built-in support for exporting PyTorch models to ONNX, so run the following command to convert our Cat/Dog model with the `onnx_export.py` script:

``` bash
python onnx_export.py --model-dir=cat_dog
```

This will create a model called `resnet18.onnx` under `jetson-inference/python/training/imagenet/cat_dog/`.

## Processing Images with TensorRT

To process some test images, we'll use the extended command-line parameters to the `imagenet-console` programs that we used previously to load our custom model:

```bash
DATASET=/media/SSD_EVO860/datasets/cat_dog

# C++

# Python
```


## Generating More Data

The images from the Cat/Dog dataset were randomly pulled from a larger <a href="https://drive.google.com/open?id=1LsxHT9HX5gM2wMVqPUfILgrqVlGtqX1o">subset of ILSCRV12</a> (22.5GB), using the [`cat-dog-dataset.sh`](../tools/cat-dog-dataset.sh) script. The images are made up of many different breeds of dogs and cats, including large felines like tigers and mountain lions since the diversity among cats was a bit lower than dogs.  Some of the images also picture humans, which the detector is essentially trained to ignore and focus on the cat vs dog content.  We purposely kept this first dataset smaller to keep the training time down, but using the script above you can re-generate the dataset with more images to create a more robust model. 

<p align="right">Next | <b><a href="pytorch-cat-dog.md">Training the Cat/Dog Dataset</a></b>
<br/>
Back | <b><a href="pytorch-transfer-learning.md">Transfer Learning with PyTorch</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>