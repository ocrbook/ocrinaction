# -*- coding: UTF-8 -*-
import numpy as np
import cv2

image = cv2.imread("../picture/lena.jpg")
# 将图像转化为灰度图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# x方向的梯度
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
# x方向梯度的绝对值
sobelX = np.uint8(np.absolute(sobelX))
# y方向的梯度
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
# y方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imwrite("../picture/sobel_x.jpg", sobelX)
cv2.imwrite("../picture/sobel_y.jpg", sobelY)
cv2.imwrite("../picture/Sobel_xy.jpg", sobelCombined)
