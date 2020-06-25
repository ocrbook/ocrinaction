# -*- coding: UTF-8 -*-
import cv2
import numpy as np
im = cv2.imread('../picture/hough.png')
edges = cv2.Canny(im, 50, 150, apertureSize=3)
result = im.copy()
minLineLength = 10
maxLineGap = 30
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite("../picture/hough_result.png", result)
