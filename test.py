# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:54:18 2018

@author: YQ
"""

from keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools

from model.cnn import ConvNet

classes = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

files = glob.glob("cifar10/test/*/*")
#model = load_model("save/cifar10.h5")
model = ConvNet.build(32,32,3,10)
model.load_weights("save/weights.h5")
lbl = pickle.load(open("labels.pkl", "rb"))

data = []
labels = []
for file in files:
    img = cv2.imread(file)
    label = file.split("\\")[-2]
    
    data.append(img)
    labels.append(label)

#%%
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)
labels = lbl.fit_transform(labels)
#%%
labels = np.argmax(labels, -1)
#%%
predictions = model.predict(data, verbose=1)
predictions = np.argmax(predictions, -1)

report = classification_report(labels, predictions)
accuracy = accuracy_score(labels, predictions)
print(report)
print("Accuracy = {:.2f}".format(accuracy))

for i in range(10):
    t = i * 1000
    pic = data[t]
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic, (150,150))
    
    plt.subplot(2, 5, i+1)
    plt.axis("off")
    plt.title(str(labels[t]) + " - " + str(predictions[t]))
    plt.imshow(pic)
    
plt.show()    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    
    
cm = confusion_matrix(labels, predictions)
plot_confusion_matrix(cm, classes)
    