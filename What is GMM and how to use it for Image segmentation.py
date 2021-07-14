# -*- coding: utf-8 -*-
"""
Created on Sat May  8 21:28:47 2021

@author: abc
"""

import numpy as np
import cv2

img = cv2.imread('BSE_Image.jpg')

#reshape our image
img2 = img.reshape((-1,3))

#define gaussian mixture model
from sklearn.mixture import GaussianMixture as GMM

#create model and fit it
gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)

#create labels and predict our image
gmm_labels = gmm_model.predict(img2)

#reconstuct image into original shape
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])

#Let's visualize the image
cv2.imwrite('segmented_BSE.jpg',segmented)

#####################################################################################################

