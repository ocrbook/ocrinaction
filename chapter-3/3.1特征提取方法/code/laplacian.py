# -*- coding: UTF-8 -*-
import numpy as np
import cv2

image = cv2.imread("../picture/lena.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imwrite("../picture/Laplacian.jpg", lap)
