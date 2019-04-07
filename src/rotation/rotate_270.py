# data set preparation file

# import necessary libraries
import glob
import os

from PIL import Image

image_list = []

for filename in glob.glob('/home/arannya/sertis/Sertis_Assignment/270/*.jpg'):
    print('-------------')
    print(filename)
    image_name = os.path.basename(filename)
    image_name = image_name.split('.')[0]
    im=Image.open(filename)
    rotated = im.rotate(270)
    print ('---- rotated -------')
    rotated.save('/home/arannya/sertis/Sertis_Assignment/training_dataset/'+image_name+'_270', 'JPEG')
