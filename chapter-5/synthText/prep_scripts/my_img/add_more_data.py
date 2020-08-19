import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image

# path to the data-file, containing image, depth and segmentation:
DB_FNAME = './dset.h5'

#add more data into the dset
more_img_file_path='./img_dir/'
more_depth_path='./depth.h5'
more_seg_path='./seg.h5'


def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path):
  db=h5py.File(DB_FNAME,'w')
  # depth_db=get_data(more_depth_path)
  # seg_db=get_data(more_seg_path)
  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')
  depth_db = h5py.File(more_depth_path,'r')
  seg_db = h5py.File(more_seg_path,'r')
  for imname in os.listdir(more_img_file_path):
    print(imname.endswith)
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print(full_path,imname)
      
      j=Image.open(full_path)
      imgSize=j.size
      # rawData=j.tostring()
      rawData = j.tobytes()
      #img=Image.fromstring('RGB',imgSize,rawData)
      img = Image.frombytes('RGB',imgSize,rawData)
      #img = img.astype('uint16')
      db['image'].create_dataset(imname,data=img)
      db['depth'].create_dataset(imname,data=depth_db[imname])
      db['seg'].create_dataset(imname,data=seg_db['mask'][imname])
      db['seg'][imname].attrs['area']=seg_db['mask'][imname].attrs['area']
      db['seg'][imname].attrs['label']=seg_db['mask'][imname].attrs['label']

  print("dset.h5 has been created.")
  db.close()
  depth_db.close()
  seg_db.close()

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path)
