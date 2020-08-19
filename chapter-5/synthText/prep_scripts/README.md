## 增加自己的图片
 将自己的背景图片放入 `/prep_script/my_img/img_dir`
  
 下载 DCNF-FCSP: https://bitbucket.org/fayao/dcnf-fcsp/get/f66628a4a991.zip 放入 `prep_scripts` 文件夹，然后重命名为`demo`.
   
  放入文件 `run_ucn.m`、`floodFill.py`、`predict_depth.m` 、`Multiscale Combinatorial Grouping`: https://github.com/jponttuset/mcg/archive/master.zip in `pre_scripts/demo`.


## 计算景深

计算图像的分割与景深，让生成的文字更加自然。
  ```
  cd prep_scripts/fayao-dcnf-new/demo

  mrun run_ucn                             //   prep_script/my_img/ucm.mat
  python floodFill.py                      //   prep_script/my_img/seg.h5

  cd ../libs/matconvnet/
  matlab -nodesktop
        >> addpath matlab/
        >> mex -setup
        >> cd matlab/
        >> vl_compilenn
        >> exit
  cd ../../demo/
  mrun predict_depth	                     //   prep_script/my_img/depth.h5   

  cd ../../my_img/
  python add_more_data.py                   //  prep_script/my_img/dset.h5

  cp dset.h5 ../../data/
  ```