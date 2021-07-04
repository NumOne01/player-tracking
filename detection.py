from __future__ import print_function
import cv2 as cv
import numpy as np
import math

backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture('output.mp4')
if not capture.isOpened():
    print('Unable to open file')
    exit(0)

n = (int)(capture.get(cv.CAP_PROP_FRAME_WIDTH))
m = (int)(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
p1 = (15, 150)
p2 = (1232, 86)
p3 = (1137, 115)
p4 = (140, 167)
points1 = np.array([p1, p2, p3, p4], dtype=np.float32)

field = cv.imread('field.png')
n = field.shape[0]
m = field.shape[1]
output_size = (n, m)
points2 = np.array([(0, 0), (n, 0), (886, 149), (162, 152)], dtype=np.float32)

H = cv.getPerspectiveTransform(points1, points2)

while True:
    field = cv.imread('field.png')
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv.warpPerspective(frame, H,  output_size)

    fgMask = backSub.apply(frame)

    # remove shadows
    ret, T = cv.threshold(fgMask, 220, 255, cv.THRESH_BINARY)
    fgMask = T

    # remove noises
    kernel = np.ones((7, 7), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    # draw players
    n, C, state, centroids = cv.connectedComponentsWithStats(fgMask)
    for i in range(n):
        if not math.isnan(centroids[i][0]):
            cv.circle(field, ((int)(centroids[i][0]), (int)(
                centroids[i][1])), 8, (255, 0, 0), 15)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('FG Mask', field)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
