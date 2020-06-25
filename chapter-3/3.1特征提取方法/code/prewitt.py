# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal

# x方向的Prewitt算子
suanzi_x = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])

# y方向的Prewitt算子
suanzi_y = np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])
image = cv2.imread("../picture/lena.jpg")
# 将图像转化为灰度图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 得到x方向矩阵
image_x = signal.convolve2d(image, suanzi_x)
# 得到y方向矩阵
image_y = signal.convolve2d(image, suanzi_y)
# 得到梯度矩阵
image_xy = np.sqrt(image_x**2 + image_y**2)
# 梯度矩阵统一到0-255
image_xy = (255.0 / image_xy.max()) * image_xy
# 保存图像
cv2.imwrite("../picture/Prewitt_x.jpg", image_x)
cv2.imwrite("../picture/Prewitt_y.jpg", image_y)
cv2.imwrite("../picture/Prewitt_xy.jpg", image_xy)
