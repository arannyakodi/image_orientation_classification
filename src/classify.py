# import necessary libraries
import os
import pickle
import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# initialize model, labeling , image , rotated angle
model_path = 'sertis_model_e100_b20.model'  # model
mlb_path = 'mlb.pickle'  # labeling
org_image = os.path.join('..', 'test_dataset/test1.jpg')  # images to be tested path
rotated_angle = 0  # final rotated angle

# load the image and resize
image = cv2.imread(org_image)
output = imutils.resize(image, width=400)

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# rotate the image to correct angle
def rotate_image_to_correct_angle(org_image, label):
    print (org_image, label)
    im = Image.open(org_image)
    image_name = os.path.basename(org_image)
    if label == 0:
        print ('0')
        rotated = im.rotate(0)
        rotated.save(image_name, 'JPEG')
    elif label == 90:
        print ('90')
        rotated = im.rotate(90)
        rotated.save(image_name, 'JPEG')
    elif label == 180:
        print ('180')
        rotated = im.transpose(Image.FLIP_TOP_BOTTOM)
        rotated.save(image_name, 'JPEG')
    elif label == 270:
        print ('270')
        rotated = im.rotate(270)
        rotated = rotated.transpose(Image.FLIP_TOP_BOTTOM)
        rotated.save(image_name, 'JPEG')


# load the trained convolutional neural network and the labelbinarizer
print("Loading network...")
model = load_model(model_path)
mlb = pickle.loads(open(mlb_path).read())

# classify the input image then find the indexes of the two class labels with the *largest* probability
print("Classifying image...")
proba = model.predict(image)[0]  # probability
idxs = np.argsort(proba)[::-1][:2]  # indexes

# loop over the indexes of the high confidence class labels and get the highest one
# TODO : can try first two suggestions as well
for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    rotated_angle = int(mlb.classes_[j])
    rotate_image_to_correct_angle(org_image, rotated_angle)
    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    print ('Label with probability : ', label)
    break
