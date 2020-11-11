import data
import models
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from tensorflow import keras

ids = data.getImageIds(0.10)
trainIds, testIds = train_test_split(ids, test_size=0.2, random_state=42)
print("Len of trainIds:", len(trainIds))
print("Len of testIds:", len(testIds))

NEWINPUTS_FOLDER = "../newinputs"
NEWOUTPUTS_FOLDER = '../newoutputs'
BACKGROUND_INPUT_FOLDER = '../newinputs'
FLOWER_INPUT_FOLDER = '../newinputs'
BACKGROUND_OUTPUT_FOLDER = '../background_output'
FLOWER_OUTPUT_FOLDER = '../flower_output'

'''
Loads background image data which a black image mask has been applied, from a list of ids
'''
def loadBackgroundImageData(ids):
    X = []
    Y = []
    for i in ids:
        img = Image.open('{}/background_{:05d}.jpg'.format(BACKGROUND_INPUT_FOLDER, i))
        img = np.array(img)

        x = rgb2lab(img)[:, :, 0]
        y = rgb2lab(img)[:, :, 1:]

        X.append(x.reshape(x.shape + (1,)))
        Y.append(y)

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y

'''
Loads flower image data which a black image mask has been applied, from a list of ids
'''
def loadFlowerImageData(ids):
    X = []
    Y = []
    for i in ids:
        img = Image.open('{}/flower_{:05d}.jpg'.format(FLOWER_INPUT_FOLDER, i))
        img = np.array(img)

        x = rgb2lab(img)[:, :, 0]
        y = rgb2lab(img)[:, :, 1:]

        X.append(x.reshape(x.shape + (1,)))
        Y.append(y)

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y

'''
Generates and saves background images. copied from data.py
'''
def generateBackgroundImages(L, AB, ids):
    dimensions = L.shape[1:3] + (3,)
    for i in range(len(ids)):
        cur = np.zeros(dimensions)
        cur[:, :, 0] = L[i][:, :, 0]
        cur[:, :, 1:] = AB[i]
        id = ids[i]
        filename = "{}/background_output{:05d}.jpg".format(BACKGROUND_OUTPUT_FOLDER, id)
        rgb = (lab2rgb(cur) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(filename)

'''
Generates and saves flower images. copied from data.py
'''
def generateFlowerImages(L, AB, ids):
    dimensions = L.shape[1:3] + (3,)
    for i in range(len(ids)):
        cur = np.zeros(dimensions)
        cur[:, :, 0] = L[i][:, :, 0]
        cur[:, :, 1:] = AB[i]
        id = ids[i]
        filename = "{}/flower_output{:05d}.jpg".format(FLOWER_OUTPUT_FOLDER, id)
        rgb = (lab2rgb(cur) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(filename)


#### training
'''
train Background and Flower individually
'''
# X_train, Y_train = loadFlowerImageData(trainIds)
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#
# Y_train /= 128
#
# model = models.getCIE94Model()
#
# model.fit(x=X_train,
#             y=Y_train,
#             batch_size=64,
#             epochs=1000)
#
# model.save("models/CIE94model_Flower_10_20_64_1000")


#### test
'''
test Background and Flower individually
'''
X_test, Y_test = loadFlowerImageData(testIds)
X_train, Y_train = loadFlowerImageData(trainIds)

model = keras.models.load_model("models/CIE94model_Flower_10_20_32_1000")

model.evaluate(X_test, Y_test, batch_size=64)
model.evaluate(X_train, Y_train, batch_size=64)
# predictions = model.predict(X_test)
# generateBackgroundImages(X_test, predictions * 127, testIds)



#### stitching predictions
# for i in testIds:
#     print(i)
#     background = Image.open('{}/background_output{:05d}.jpg'.format(BACKGROUND_OUTPUT_FOLDER, i))
#     flower = Image.open('{}/flower_output{:05d}.jpg'.format(FLOWER_OUTPUT_FOLDER, i))
#     mask = Image.open('{}/mask_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))
#     dst = Image.composite(background, flower, mask)
#     dst.save('{}/stitched{:05d}.jpg'.format(NEWOUTPUTS_FOLDER, i))


#### generate prediction for any image
# TODO
# background = Image.open('{}/background_output{:05d}.jpg'.format(BACKGROUND_OUTPUT_FOLDER, i))
# flower = Image.open('{}/flower_output{:05d}.jpg'.format(FLOWER_OUTPUT_FOLDER, i))
# mask = Image.open('{}/mask_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))
# dst = Image.composite(background, flower, mask)
# dst.save('{}/stitched{:05d}.jpg'.format(NEWOUTPUTS_FOLDER, i))