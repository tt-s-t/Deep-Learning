# LeNet

## own folder
Build LeNet by numpy. It was used to classify the mnist data set.But it does not speed up, so it will run slowly. It is only for detailed understanding of the principle. It is better to use the built-in function or Lenet normally.

***layer_linear_relu_logsoftmax.py***
The construction of logsoftmax, linear and relu.

***layer_conv_pool_flatten.py***
The construction of convolute layer, pool layer and flatten layer.

***lenet_model.py***
The whole model construction process, including training, testing, data set processing, and so on.

***lenet_own_run.py***
Load data and call the previously defined model class for training and testing.

## lenet_pytorch.py
Use the modules in torch to build a lenet to realize network training and testing. At the same time, visdom is used to visually see the change curve of loss in real time.
This code refers to "https://github.com/activatedgeek/LeNet-5.git".