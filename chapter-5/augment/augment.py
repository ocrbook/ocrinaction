# auther:liushuchun
import cv2
import numpy as np
import sys
import os


if sys.version_info.major == 2:
    print("python2 excute")
elif sys.version_info.major == 3:
    print("python3 excute")


def rotate(img, angle, center=None, scale=1.0):
    # get the dimension of the img
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(
        img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return rotated_img


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    # get the dimension of the img

    dm = None

    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width:
        r = width / float(w)
        dm = (width, int(h * r))
    else:
        r = height / float(h)
        dm = (int(w * r), height)

    resized_img = cv2.resize(img, dm, interpolation=inter)

    return resized_img


def adjust_brightness_contrast(img, brightness=0., contrast=0.):
    """
    Adjust the brightness or contrast of image
    """
    beta = 0
    return cv2.addWeighted(img, 1 + float(contrast) / 100., img, beta, float(brightness))


def blur(img, typ="gaussian", kernal=(2, 2)):
    """
    Blur the image
    :params:
            typ: "gaussian" or "median"
    """
    if typ == "gaussian":
        return cv2.GaussianBlur(img, kernal, 0, None, 0)
    elif typ == "median":
        return cv2.blur(img, kernal)
    else:
        return img


if __name__ == "__main__":
    img = cv2.imread("sample.png")
    (h, w) = img.shape[:2]
    blur_img = blur(img, kernal=(5, 5))
    blur_img = blur(blur_img, kernal=(3, 3))
    cv2.imwrite("blured.png", blur_img)
    cv2.imwrite("src.png", img)

    decre_contrasted_img = adjust_brightness_contrast(
        img, brightness=0.5, contrast=-50)
    cv2.imwrite("decre_contrasted.png", decre_contrasted_img)
    incre_contrasted_img = adjust_brightness_contrast(
        img, brightness=1.0, contrast=50)
    cv2.imwrite("incre_contrasted.png", incre_contrasted_img)

    resized_long_img = resize(img, width=int(w * 1.4), height=h)
    cv2.imwrite("resized_long_img.png", resized_long_img)

    resized_short_img = resize(img, width=w, height=int(0.5 * h))
    cv2.imwrite("resized_short_img.png", resized_short_img)

    rotated_img = rotate(img, angle=5)
    cv2.imwrite("roated_img.png", rotated_img)
