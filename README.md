# 교육 목표

딥러닝의 개념을 이해하고 Keras로 실제 딥러닝 코드를 구현하고 실행시킬 수 있다.


# 대상

python으로 기본적인 프로그래밍을 할 수있는 개발자 혹은 개발 희망자.


# 교육 상세 목표

다음의 코드를 다룰 수 있는 것을 목표로 함
```
# copy from https://www.tensorflow.org/beta/tutorials/keras/basic_classification

!pip install -q tensorflow==2.0.0-beta1

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))




plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()




model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)




test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])

```


# 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP notebook
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        op
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
        to numpy
    matplot
        draw line graph
        scatter
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tunning
            overfitting, underfitting
            regularization, dropout, batch normalization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            varnilla, VGG16
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
        target : text/image

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularier
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```


# 일자별 계획

## 1일차

- 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)
- DNN in Keras : [dnn_in_keras.ipynb](material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/dnn_in_keras.ipynb)


<br>

## 2일차

- DNN in Keras : [dnn_in_keras.ipynb](material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/dnn_in_keras.ipynb)
- DNN
    - 분류기로서 DNN : [dnn_as_a_classifier.ipynb](material/deep_learning/dnn_as_a_classifier.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/dnn_as_a_classifier.ipynb)
    - IRIS 분류: [dnn_iris_classification.ipynb](material/deep_learning/dnn_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/dnn_iris_classification.ipynb)
    - MNIST 분류 : [dnn_mnist.ipynb](material/deep_learning/dnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/dnn_mnist.ipynb)
- 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
- CNN
    - MNIST 영상분류 : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/cnn_mnist.ipynb)
    - CIFAR10 컬러영상분류 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/cnn_cifar10.ipynb)
    - IRIS 분류 : [iris_cnn.ipynb](material/deep_learning/iris_cnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/iris_cnn.ipynb)


<br>

## 3일차
- VGG로 영상 분류, 전이학습 : [VGG16_classification_and_cumtom_data_training.ipynb](material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)
- 알파고 이해하기 : [understanding_ahphago.pptx](material/deep_learning/understanding_ahphago.pptx)
- 영역 분할(segmentation) - U-Net : [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/unet_segementation.ipynb)
- 얼굴 인식(face recognition) : [Face_Recognition.ipynb](material/deep_learning/Face_Recognition.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/Face_Recognition.ipynb)
- GAN 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx), [wgan_pg_mnist.ipynb.ipynb](material/deep_learning/wgan_pg_mnist.ipynb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/wgan_pg_mnist.ipynb.ipynb)
- RNN 이해하기
- 강화학습 이해하기

<br>

## 4일차
- text 분류 - CNN, RNN 사용한 : [text_classification.ipynb](material/deep_learning/text_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/text_classification.ipynb)
- RNN을 사용한 MNIST 분류 : [mnist_rnn.ipynb]](material/mnist_rnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/mnist_rnn.ipynb)
- 물체 탐지(Object Detection) 
    - YOLO : [object_detection.md](material/deep_learning/object_detection.md)
    - RetinaNet : [object_detection_retinanet.ipynb](material/deep_learning/object_detection_retinanet.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/object_detection_retinanet.ipynb)
    - annotation tool : https://github.com/virajmavani/semi-auto-image-annotation-tool
- Keras Functional API : [functional_api.ipynb](material/functional_api.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/wiset_2020_06/blob/master/material/deep_learning/functional_api.ipynb)
- 딥러닝 개발 계획 리뷰


<br>

## 5일차
- 논문 리뷰와 논문 재현 : [의학 논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit)
- Kaggle 문제 풀이 리뷰
    - Intel Image Classification https://www.kaggle.com/puneet6060/intel-image-classification/kernels
		- https://www.kaggle.com/uzairrj/beg-tut-intel-image-classification-93-76-accur
		- CNN, Keras : https://www.kaggle.com/vincee/intel-image-classification-cnn-keras
		- Dense-Net : https://www.kaggle.com/youngtaek/intel-image-classification-densenet-with-focalloss

    - COVID19 Global Forecasting https://www.kaggle.com/c/covid19-global-forecasting-week-3
        - sigmoid model : https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april
        - linear model : https://www.kaggle.com/aerdem4/covid-19-basic-model-not-leaky
- 딥러닝 채용 공고 리뷰


<br>

## 기타

교육환경, numpy, pandas, matplot
- 교육 환경 : [env.md](material/env.md)
- numpy : 데이터 로딩, 보기, 데이터 변환, 형태 변경 : [library.md](material/library.md)
- linux 기본 명령어 : 
    - bash, cd, ls, rm, mkdir, mv, tar, unzip
    - docker, pip, apt, wget, EVN, git
    - 교육 자료
        - [linux.md](material/linux.md)
        - [linux_exercise.md](material/linux_exercise.md)
- pandas, matplot : [ibrary.md](material/library.md)

기타 자료
- Cycle GAN 리뷰 : [cycle_gan.pdf](material/cycle_gan.pdf)
- 흥미로운 딥러닝 결과 : [some_interesting_deep_learning.pptx](material/deep_learning/some_interesting_deep_learning.pptx)
- yolo를 사용한 실시간 불량품 탐지 사례 : [yolo_in_field.mp4](material/deep_learning/yolo_in_field.mp4)
- GAN을 사용한 생산설비 이상 탐지 : [anomaly_detection_using_gan.pptx](material/deep_learning/anomaly_detection_using_gan.pptx)
- 이상탐지 동영상 : [drillai_anomaly_detect.mp4](material/deep_learning/drillai_anomaly_detect.mp4)


