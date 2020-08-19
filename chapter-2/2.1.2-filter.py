import numpy as np
import cv2

########     四个不同的滤波器    #########
img = cv2.imread('img/lena_noise.png')
# 平滑线性滤波滤波
img_mean = cv2.blur(img, (5, 5))
cv2.imwrite("mean.png",img_mean)
# 高斯滤波
img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite("Guassian.png",img_Guassian)
# 中值滤波
img_median = cv2.medianBlur(img, 5)
cv2.imwrite("median.png",img_median)
# 双边滤波
img_bilater = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imwrite("bilater.png",img_bilater)
