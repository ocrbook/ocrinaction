# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 直接读为灰度图像
img = cv2.imread('../picture/fuliye.png', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 先取绝对值，表示取模。取对数，将数据范围变小
magnitude_spectrum = 20 * np.log(np.abs(fshift))
cv2.imwrite("../picture/original.jpg", img)
cv2.imwrite("../picture/center.jpg", magnitude_spectrum)
