# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 08:48:29 2018

@author: YQ
"""

# import required libraries
import cv2
import glob 
import numpy as np
import matplotlib.pyplot as plt

from model.cnn import ConvNet

from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import pickle

# initialize variables
files = glob.glob("cifar10/train/*/*")    # collect all img filenames as list
size = 32
n_channels = 3

# hyperparameters
EPOCHS = 5
BS = 256
LR = 1e-3

labels = []
data = []
# read image into a list -> data
# create list of labels for each image -> labels
for file in files:
    img = cv2.imread(file)
    label = file.split("\\")[-2]
    
    data.append(img)
    labels.append(label)
    #%%
# why divide 255?
data = np.array(data, dtype=np.int32)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

labels = np.array(labels)
n_class = len(np.unique(labels))
lbl = LabelBinarizer()
labels = lbl.fit_transform(labels)
pickle.dump(lbl, open("labels.pkl", "wb"))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=9)

print("compiling model...")
model = ConvNet.build(size, size, n_channels, n_class)
model.load_weights("save/weights.h5")
model.summary()
opt = Adam(lr=LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# image augmentation, dataset is small
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         fill_mode="nearest")
print("training SmallVGG...")
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=BS), 
                         validation_data=(X_test, y_test),
                         epochs=EPOCHS, verbose=1)

#model.save_weights("save/weights.h5")
#model.save("save/cifar10.h5")

# plot the loss and acc into a graph
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")