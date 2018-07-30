# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:57:34 2018

@author: YQ
"""

import pickle
import cv2
import os

labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

for l in labels:
   os.mkdir("train\\"+l)
   os.mkdir("test\\"+l)

X_train = pickle.load(open("X_train.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_train = pickle.load(open("y_train.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

for i in range(10):
    temp = X_train[y_train==i]
    for z, pic in enumerate(temp):
        filename = "train\\" + labels[i] + "\\" + str(z).zfill(5) + ".png"
        cv2.imwrite(filename, cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        
        filename = "test\\" + labels[i] + "\\" + str(z).zfill(5) + ".png"
        cv2.imwrite(filename, cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        