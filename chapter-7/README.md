### 环境

```
Linux (Windows 暂未获得官方支持)
Python 3.7
PyTorch 1.6 或更高版本
torchvision 0.7.0
CUDA 10.1
NCCL 2
GCC 5.4.0 或更高版本
```

### 下载代码

给大家准备了一个小的基础框架。

```bash
git clone git@github.com:ocrbook/ToyOCR.git
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 准备数据

将数据整理成coco格式

### 训练模型

```bash
sh train_toydet.sh
```
