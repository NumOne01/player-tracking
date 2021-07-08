import cv2 as cv
import numpy as np
from tensorflow import keras

PATCH_WIDTH = 30
PATCH_HEIGHT = 45
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
RED_CLASS = 0
BLUE_CLASS = 1


def getPlayerColor(patch):
    patch = np.expand_dims(patch, axis=0)
    prediction = model.predict(patch)
    return RED_COLOR if np.argmax(
        prediction[0]) == RED_CLASS else BLUE_COLOR


def getPatch(contour):
    x, y, w, h = cv.boundingRect(contour)
    patch = frame[y:y+h, x:x+w]
    patch = cv.resize(patch, (PATCH_WIDTH, PATCH_HEIGHT))
    return patch


def savePatch(patch, index):
    if patch.shape[0] > 0 and patch.shape[1] > 0:
        cv.imwrite('data/pic_' + str(index) + '.jpg',
                   patch)


backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture('output.mp4')

if not capture.isOpened():
    print('Unable to open file')
    exit(0)

field = cv.imread('field.jpg')
n = field.shape[0]
m = field.shape[1]

output_size = (m, n)
points1 = np.array([(1280, 745),  # br
                    (1238, 87),  # tr
                    (9, 149),  # tl
                    (0, 855)]).astype(np.float32)  # bl

points2 = np.array([(375, 380),  # br
                    (578, 90),  # tr
                    (45, 90),  # tl
                    (250, 380)]).astype(np.float32)  # bl

H = cv.getPerspectiveTransform(points1, points2)
model = keras.models.load_model('model')

i = 0
j = 0
while True:
    field = cv.imread('field.jpg')
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv.warpPerspective(frame, H,  output_size)

    fgMask = backSub.apply(frame)

    # remove shadows
    ret, fgMask = cv.threshold(fgMask, 254, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(
        fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if(cnt[0][0][1] < 70):
            continue
        area = cv.contourArea(cnt)
        # Remove very small contours
        if area > 25:
            x = (int)(cnt[0][0][0])
            y = (int)(cnt[0][0][1])
            patch = getPatch(cnt)
            player_color = getPlayerColor(patch)
            cv.circle(field, (x, y), 4, player_color, 5)
            # save patches every 30 frames
            if j % 30 == 0:
                savePatch(patch, i)
                i += 1
    j += 1

    cv.imshow('Field', field)
    cv.imshow('Frame', frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 113:
        break
