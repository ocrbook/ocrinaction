# -*- coding: UTF-8 -*-
import cv2
image = cv2.imread("../picture/lena.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Canny边缘检测
canny = cv2.Canny(image, 30, 150)
cv2.imwrite("../picture/Canny.jpg", canny)
