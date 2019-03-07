# Neural Network for character recognition

This repository consists of implementation of neural network for handwritten characters recognition.
Model is trained on subset of MNIST dataset [available here](http://yann.lecun.com/exdb/mnist/).

## Prerequisites

List of used packages:
* numpy
* pickle
* scipy

In the first draft train,py was also using histrogram.py for feature extraction prepared by Jean KOSSAIFI <jean.kossaifi@gmail.com>
This library is not used in final version.

## Trainig

Size of the hidden layer of the network:

```
hidden_size=55
```

Size of output layer corresponds to number of characters that model will recognize:

```
output_size=36
```

Initial parameters of the model are generated randomly:

```
w1= np.random.uniform(low=-0.1, high=0.1, size=(hidden_size,(D+1)) )
w2= np.random.uniform(low=-0.1, high=0.1, size=(output_size, (hidden_size+1)))
```
In testing this proved to give better result than startin with weights of 0 and 1.

Model is evaluated on the basis of classification error.
Final parameters are saved in w1.txt and w2.txt file.

## Prediction

Prediction is done using predict function in predict.py, producing array of numbers {0, ..., 35} corresponding to characters.
Final model achieved accuracy of 85%.
