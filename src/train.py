# import necessary libraries
import glob
import os
import pickle
import cv2
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from model.smallervggnet import SmallerVGGNet

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100  # number of epochs
INIT_LR = 1e-3  # learning rate
BS = 20  # batch size
IMAGE_DIMS = (96, 96, 3)  # image dimensions

# initialize the data and labels
dataset_dir = os.path.join('..', 'training_dataset/')
data = []
labels = []


# get images and labels
def getImages_and_labels(dataset):
    image_data = []
    labels = []
    print (dataset)
    path = dataset + "*"
    for file in glob.glob(path):
        image_name = os.path.basename(file)
        label = image_name.split('_')[1]
        labels.append(label)
        image = cv2.imread(file)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        image_data.append(image)
    return image_data, labels


# assigning derived data and labels
data, labels = getImages_and_labels(dataset_dir)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# convert labels into numpy arrays
labels = np.array(labels)

print("data matrix: {} images ({:.2f}MB)".format(
    len(labels), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special labelbinarizer implementation
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)
print ('Labels : ', labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print ('i', i, 'label', label, 'mbclass', mlb.classes_)

# divide the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform  classification
print("Compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
                            finalAct="sigmoid")

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("Training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)

# save the model to disk
print("Serializing network...")
model.save('sertis_model_e100_b20.model')

# save the label binarizer to disk
print("Serializing label binarizer...")
f = open('mlb.pickle', "wb")
f.write(pickle.dumps(mlb))
f.close()
