'''
Trains a model and saves it into the models folder.
Also saves the train test ID split into train.npy and test.npy so that the test script
    knows which IDs to test on.
'''

import data
import models
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ids = data.getImageIds(0.502)
np.random.seed(42)
ids = np.random.choice(ids, len(ids) // 5 + 1)
trainIds, testIds = train_test_split(ids, test_size=0.2, random_state=42)
testIds = testIds[:-1]
#trainIds = trainIds[:-51]
print("Len of trainIds:", len(trainIds))
print("Len of testIds:", len(testIds))

np.save("npy/train_502_160_40_200", trainIds)
np.save("npy/test_502_160_40_200", testIds)

X_train, Y_train = data.loadImageData(trainIds)
# Model specific code

# MSE model
'''
Y_train /= 128
model = models.getMSEModel()

model.fit(x=X_train, 
    y=Y_train,
    batch_size=32,
    epochs=100)

model.save("models/MSEmodel_50_20_32_100")
'''

Y_train /= 128
model = models.getCIE94Model(40)

model.fit(x=X_train,
    y=Y_train,
    batch_size=40,
    epochs=200)

model.save("models/CIEmodel_502_160_40_200")

'''
# Classification model

bins = np.load("npy/pts_in_hull.npy")
model = models.getClassificationModel()

# Manual loop if unable to process all in RAM at once
epochs = 100
batch_size = 4
for i in range(epochs):
    print("Epoch {}/{}:".format(i + 1, epochs))
    batchIds = [trainIds[i * batch_size : (i + 1) * batch_size] for i in range(math.ceil(len(trainIds) / batch_size))]
    b = 1
    for ids in batchIds:
        print("Epoch {}/{}, Batch {}/{}".format(i + 1, epochs, b, len(batchIds)))
        b += 1
        X_train, Y_train = data.loadImageData(ids)
        Y_train = data.batchQuantize(Y_train, bins)

        model.fit(x=X_train, 
            y=Y_train,
            batch_size=batch_size,
            epochs=1)

model.save("models/ClassificationModel_50_20_4_100")
'''
