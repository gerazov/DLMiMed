# DLMiMed
Deep learning for breast tumour classification based on medical microwave imaging.

This repository contains the code used to classify the tumour shape using backscatter signals obtained using FDTD numerical simulation of 3D tumour models embedded in homogeneous adipose breast tissue. The analysis and results have been reported in the paper:

Gerazov B. and R.C. Conceição, “Deep learning for tumour classification in homogeneous breast tissue in medical microwave imaging,” IEEE EUROCON 2017, Ohrid, Macedonia,  6-8 Jul 2017.

The code combines Theano and scikit-learn to perform a k-fold crossvalidation performance analysis of deep learning including deep neural networks (in `dnn.py`) and convolutional neural networks (in `cnn.py`). The code also includes a SVM trained on the features obtained from the penultimate layer of the DNN (in `svm.py`). The utility functions used to build the neural networks are stored in `dlutils.py`. Although the dataset used in the paper is not provided it is our hope that the code can be useful for other researchers implementing crossvalidation performance analysis of deep learning architectures by combining Theano and scikit-learn.

The DNN part is largely based on [Theano's tutorial](http://deeplearning.net/software/theano/tutorial/examples.html) and on the `deeplearning.net` [tutorials](http://deeplearning.net/tutorial/).

All the code is distributed with the GNU General Public License v.3, given in `LICENSE`.


Branislav Gerazov

Departement of Electronics

[Faculty of Electrical Engineering and Information Technologies](http://feit.ukim.edu.mk)

[Ss Cyril and Methodius University of Skopje](http://ukim.edu.mk/)

