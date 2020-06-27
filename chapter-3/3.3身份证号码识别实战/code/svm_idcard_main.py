# -*- coding: UTF-8 -*-
import cv2
import os
import numpy as np
from sklearn import svm
from PIL import Image


# 垂直投影


def verticle_projection(thresh1):
    (h, w) = thresh1.shape
    a = [0 for z in range(0, w)]
    # 记录每一列的波峰
    # 遍历一列
    for j in range(0, w):
        # 遍历一行
        for i in range(0, h):
            # 如果改点为黑点
            if thresh1[i, j] == 0:
                # 该列的计数器加一计数
                a[j] += 1
                # 记录完后将其变为白色
                thresh1[i, j] = 255
    # 遍历每一列
    for j in range(0, w):
        # 从该列应该变黑的最顶部的点开始向最底部涂黑
        for i in range((h - a[j]), h):
            # 涂黑
            thresh1[i, j] = 0
    # 存储所有分割出来的图片
    roi_list = list()
    start_index = 0
    end_index = 0
    in_block = False
    for i in range(0, w):
        if in_block == False and a[i] != 0:
            in_block = True
            start_index = i
        elif a[i] == 0 and in_block:
            end_index = i
            in_block = False
        roiImg = thresh1[0:h, start_index:end_index + 1]
        roi_list.append(roiImg)
    return roi_list


# 将二值化后的数组转化成网格特征统计图


def get_features(array):
    # 拿到数组的高度和宽度
    h, w = array.shape
    data = []
    for x in range(0, w / 4):
        offset_y = x * 4
        temp = []
        for y in range(0, h / 4):
            offset_x = y * 4
            # 统计每个区域的1的值
            sum_temp = array[0 + offset_y:4 +
                                          offset_y, 0 + offset_x:4 + offset_x]
            temp.append(sum(sum(sum_temp)))
        data.append(temp)
    return np.asarray(data)


def train_main():
    # 读取训练样本
    train_path = "../dataset/train/"
    train_files = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_X = []
    train_y = []
    for train_file in train_files:
        pictures = os.listdir(train_path + train_file)
        for picture in pictures:
            img = cv2.imread(train_path + train_file + "/" + picture)
            img = cv2.resize(img, (32, 32))
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(
                gray_image, 130, 255, cv2.THRESH_BINARY)
            feature = get_features(thresh1)
            feature = feature.reshape(feature.shape[0] * feature.shape[1])
            train_X.append(feature)
            train_y.append(train_file)
            train_X = np.array(train_X)
            train_y = np.array(train_y)
    linearsvc_clf = svm.LinearSVC()
    linearsvc_clf.fit(train_X, train_y)
    return linearsvc_clf


def test_main(linearsvc_clf):
    # 原图
    img = cv2.imread("../dataset/test/idcard1.jpg")
    # 灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值图
    ret, thresh1 = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
    # 垂直投影
    roi_list = verticle_projection(thresh1)
    test_X = []
    # 输入分类器
    for single in roi_list:
        single = cv2.resize(single, (32, 32), interpolation=cv2.INTER_CUBIC)
        feature = get_features(single)
        feature = feature.reshape(feature.shape[0] * feature.shape[1])
        test_X.append(feature)
    test_X = np.array(test_X)
    result = linearsvc_clf.predict(test_X)
    print(result)


if __name__ == '__main__':
    linearsvc_clf = train_main()
    test_main(linearsvc_clf)
