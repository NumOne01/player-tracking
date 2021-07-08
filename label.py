import pickle
import cv2
from detection import BLUE_CLASS, RED_CLASS

images = []
for i in range(0, 2000):
    images.append('data/pic_' + str(i) + '.jpg')

samples = []
labels = []

for index, image in enumerate(images):
    image = cv2.imread(image)
    cv2.imshow('pic', image)
    keyboard = cv2.waitKey()
    cv2.destroyWindow('pic')
    print(keyboard)
    if keyboard == 113:  # q key
        break
    if keyboard == 32:  # space key
        continue
    if keyboard == 114:   # r key
        samples.append(image)
        labels.append(RED_CLASS)
        continue
    if keyboard == 98:       # b key
        samples.append(image)
        labels.append(BLUE_CLASS)
data = (samples, labels)
file = open("labeld_data", "wb")
pickle.dump(data, file)
file.close()
