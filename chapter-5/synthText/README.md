# 主要依赖：
- python3 
- OpenCV3
  
相较于原版，主要修改如下：
  1.从python2到python3支持
  2.支持中文
  3.opencv2改为opencv3
  4.修复了一些错误

一般而言：开源的数据集已经足够用于文字检测项目，所以生成更多应用在文字识别阶段，这里只是作为演示教学使用。

# 安装

```
pip3 install -r requirements.txt
```

### 生成数据

```
python gen.py --viz
```

  - **dset.h5**: 里面有5张图片，可以下载其他图片
  - **data/fonts**: 一些字体
  - **data/newsgroup**: 一些语料
  - **data/models/colors_new.cp**: Color模型
  - **data/models**:模型相关
生成的结果放在results当中。

```
python visualize_results.py
```
### 预生成的数据
80w张生成的数据[链接](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### [update] 增加新的背景图片
 [here](https://github.com/ankush-me/SynthText/tree/master/prep_scripts).

* `predict_depth.m` 使用dcnf-fcsp网络生成网络的深度信息 [Liu etal.](https://bitbucket.org/fayao/dcnf-fcsp/) 也可以使用 (e.g., [this](https://github.com/iro-cp/FCRN-DepthPrediction)) 效果更好
* `run_ucm.m` and `floodFill.py` 获得segment信息 [gPb-UCM](https://github.com/jponttuset/mcg).


