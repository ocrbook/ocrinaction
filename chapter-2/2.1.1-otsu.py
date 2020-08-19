import cv2
from matplotlib import pyplot as plt

image = cv2.imread("img/2-1.png")
# 将输入图像转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 绘制灰度图
plt.subplot(311), plt.imshow(gray, "gray")
plt.title("input image"), plt.xticks([]), plt.yticks([])
# 对灰度图使用 Ostu 算法
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# 绘制灰度直方图
plt.subplot(312), plt.hist(gray.ravel(), 256)
# 标注 Ostu 阈值所在直线
plt.axvline(x=ret1, color='red', label='otsu')
plt.legend(loc='upper right')
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
# 绘制二值化图像
plt.subplot(313), plt.imshow(th1, "gray")
plt.title("output image"), plt.xticks([]), plt.yticks([])
plt.show()
