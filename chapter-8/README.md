# OCR实践 第八章 深度学习识别模型 CRNN
本目录主要为基于深度学习框架pytorch实现的crnn模型，介绍模型的论文地址为：[An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf); 目前本代码项目可以在pytorch的1.0版本中成功运行。



## 安装环境
- 系统：Ubuntu 16.04
- 显卡加速库：Cuda 10.0版本
- python: 版本>=3.5
- 深度学习框架：pytorch 1.0版本
- 其他可能必要的python依赖库，如numpy，cv2等


## 训练
#### 训练命令

```bash
python3 train.py --trainroot {训练集根目录} --valroot {验证集根目录}
```

- 以ICDAR2013为例
```bash
python3 train.py \
    --trainroot ./data/Challenge2_Training_Task3_Images_GT/ \
    --valroot ./data/Challenge2_Training_Task3_Images_GT/ \
    --Adam
```

#### 数据格式
- 标注文件可以统一命名为 “gt.txt”，训练命令中根目录是指可以直接访问到“gt.txt”的目录。
- gt.txt的组织格式如下：
```bash
{图片与根目录的相对路径}, {图片对应的字符串}

示例：
img_1.jpg, aaa
img_2.jpg, aaa
img_3.jpg, aaa
...
此时gt.txt与图片在相同根目录下，以此类推...
```

#### 训练数据集
- 本项目提供两个数据集供读者进行调试：

```bash
ICDAR13(纯英文): (度盘链接) https://pan.baidu.com/s/1zp7oytJdqIYVhHnHj6ja3Q 提取码：r22p
ICPR-MTWI(中英文): (度盘链接) https://pan.baidu.com/s/1zyR96sOXYSWZz079B9y5Vg 提取码：stv3 
```

## 测试
- 本文在demo.py中提供了简单的测试代码，读者可以自行尝试修改以测试自己的模型。