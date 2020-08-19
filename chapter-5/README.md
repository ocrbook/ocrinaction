
## 本章主要分为：
   - augment
   - synthText
   - 数据集

目录：
- [本章主要分为：](#本章主要分为)
- [一些好用的OCR数据集](#一些好用的ocr数据集)
  - [1、ICDAR2017-RCTW-17](#1icdar2017-rctw-17)
  - [2、阿里巴巴天池大赛数据集](#2阿里巴巴天池大赛数据集)
  - [3、ICDAR2019-LSVT](#3icdar2019-lsvt)
  - [4、 其他数据集](#4-其他数据集)
  - [5、Synthtext仿真数据](#5synthtext仿真数据)
  - [6、中文文档文字识别](#6中文文档文字识别)
  - [7.一些字体](#7一些字体)
  - [8.一些语料](#8一些语料)

除此之外，这里罗列补充一些数据集方便大家下载使用。
## 一些好用的OCR数据集
常用的数据集，一般而言，会作为模型的finetune数据集，作为模型的基底，而后增加自己的业务数据训练会得到更鲁棒的效果。

### 1、ICDAR2017-RCTW-17
- 数据源：https://rctw.vlrlab.net/
- 语言： 中英文等
- 数据简介：：ICDAR 2017-RCTW(Reading Chinest Text in the Wild)，由Baoguang Shi等学者提出。RCTW主要是中文，共12263张图像，其中8034作为训练集，4229作为测试集，标注形式为四点标注， 数据集绝大多数是相机拍的自然场景，一些是屏幕截图；包含了大多数场景，如室外街道、室内场景、手机截图等等。
- 下载地址：https://pan.baidu.com/s/1RTDcrf5HCCKtzgNeObrRiA 提取码: xjxu

### 2、阿里巴巴天池大赛数据集
- 数据源：https://tianchi.aliyun.com/competition/entrance/231685/introduction
- 语言： 中英文
- 数据简介：商品类图片的文字检测，主要有8034张训练图片和4229张测试图片。
- 下载地址: https://pan.baidu.com/s/1q1BX0vgVY6YtCixo_YajBw 提取码: 4gvh

### 3、ICDAR2019-LSVT
- 数据源：https://rrc.cvc.uab.es/?ch=16
- 数据介绍： 由baidu公司在19年icdar发布，共45w中文街景图像，包含5w（2w测试+3w训练）全标注数据（文本坐标+文本内容），40w弱标注数据（仅文本内容），其中，test数据集的label没有开源.如图所示：
  - ![](./images/LSVT.jpg)
  - ![](./images/LSVT_unlabeled.jpg)
- 下载地址：https://pan.baidu.com/s/13iDBtyKYE37qM9MPm13AvA 提取码: u6td

### 4、 其他数据集
- ReCTS数据集:包括25,000张带标签的图像，训练集包含20,000张图像，测试集包含5,000张图像。这些图像是在不受控制的条件下通过电话摄像机野外采集的。它主要侧重于餐厅招牌上的中文文本。 数据集中的每个图像都用文本行位置，字符位置以及文本行和字符的成绩单进行注释。用具有四个顶点的多边形来标注位置，这些顶点从左上顶点开始按顺时针顺序排列。
- CDAR2019-ArT数据集：该数据集共含10,166张图像，训练集5603图，测试集4563图。由Total-Text、SCUT-CTW1500、Baidu Curved Scene Text (ICDAR2019-LSVT部分弯曲数据) 三部分组成，包含水平、多方向和弯曲等多种形状的文本。

由于弯曲文本应用比较少，所以这里不放出具体的链接，读者可以自行到icdar竞赛官网下载:https://rrc.cvc.uab.es/?ch=12&com=introduction

### 5、Synthtext仿真数据
- 数据源:
- 数据集介绍：是由Synthtext仿真程序生成
- 下载地址：http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip

### 6、中文文档文字识别
- 数据类型：识别类数据
- 数据介绍：主要生成280X32尺寸大小的灰度图片，用于识别，增加了各种仿射变换和特效
- 数据源：https://github.com/YCG09/chinese_ocr  
- 链接: https://pan.baidu.com/s/1oFHbfUTC6CQbqjgxOe3IAQ 提取码: gyux

### 7.一些字体
- **字体简介**:一些能用的字体,用于生成识别仿真数据。
- 链接: https://pan.baidu.com/s/1oir-3obLlBxvEA50cBTjSw 提取码: v9c4

### 8.一些语料
- 简介：在识别阶段需要一些语料，这里选用wiki百科和新闻数据集作为语料库。
- 链接1：http://p3gr9bd3t.bkt.clouddn.com/corpus.zhwiki.simplified.txt
- 链接2：http://p3gr9bd3t.bkt.clouddn.com/corpus.zhwiki.simplified.txt