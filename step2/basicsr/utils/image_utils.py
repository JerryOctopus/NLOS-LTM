import torch
import numpy as np
import pickle
from PIL import Image
import cv2
import pdb

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def load_img(filepath):
    # cv2.imread()读入BGR格式，0-255范围，cv2.cvtColor(p1-需要转换的图片,p2-转换成何种格式) 颜色空间转换函数

    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB) # BGR转换为RGB
    img = img.astype(np.float32)
    img = cv2.resize(img,(256,256))
    #img = cv2.resize(img,(224,224))
    
    
    img = img/255.
    return img





