# -*- coding: UTF-8 -*-
import cv2
from datetime import datetime
import numpy as np

np.set_printoptions(suppress=True)


def my_humoments(img_gray):
    moments = cv2.moments(img_gray)
    humoments = cv2.HuMoments(moments)
    # 取对数
    humoments = np.log(np.abs(humoments))
    print(humoments)


if __name__ == '__main__':
    t1 = datetime.now()
    fp = '../picture/lena.jpg'
    img = cv2.imread(fp)
    # 缩放
    h, w, _ = img.shape
    img = cv2.resize(img, (h / 2, w / 2), cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../picture/scale.jpg", img_gray)
    # 旋转
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../picture/rotate.jpg", img_gray)
    # 垂直镜像
    img = cv2.flip(img, 0, dst=None)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../picture/flip.jpg", img_gray)
    my_humoments(img_gray)
