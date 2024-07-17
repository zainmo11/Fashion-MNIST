# Fashion-MNIST Classification with dlib C++
## Overview
This project demonstrates how to use the Fashion-MNIST dataset for image classification using dlib's C++ library. The neural network is built with dlib's deep learning tools and utilizes CUDA for acceleration.

## Accuracy
The trained model achieved an accuracy of approximately 88% on the test set.

## Dataset
The Fashion-MNIST dataset contains grayscale images of fashion items. Each image is 28x28 pixels, and there are 10 classes (categories) of clothing items. The dataset is split into training and test sets.

- Training set: 60,000 images
- Test set: 10,000 images
- The dataset can be downloaded from [Kaggle's Fashion-MNIST dataset page](https://www.kaggle.com/datasets/zalando-research/fashionmnist/).

## Requirements
- dlib: Install dlib from [dlib's official website](http://dlib.net/) or via package managers.
- CUDA: Ensure you have a compatible CUDA version installed for GPU acceleration.
- C++ compiler: Ensure your C++ compiler supports C++11 or later.
## Setup
- Download the dataset: Download fashion-mnist_train.csv and fashion-mnist_test.csv from Kaggle and place them in your working directory.
- Include dlib and CUDA: Ensure that dlib is correctly installed and configured to use CUDA.



## How to Compile and Run
Compile: Use a C++ compiler that supports C++11 or later and link with dlib and CUDA libraries.

``` bash
g++ -std=c++11 -O3 -I/path/to/dlib -L/path/to/dlib/lib -ldlib -lcuda -lcudart -o fashion_mnist_classifier main.cpp

```
Run: Execute the compiled binary.
``` bash

./fashion_mnist_classifier
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.
