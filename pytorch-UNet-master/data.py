import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from time import time
import os
from skimage.io import imread
import copy
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from torchvision.models import vgg19
from torchsummary import summary
from PIL import Image

classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','potted plant','sheep','sofa','train','tv/monitor']
colormap = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]]
def image2label(image,colormap):
    cm2lbl = np.zeros(256**3)
    print(cm2lbl)
    for i,cm in enumerate(colormap):
        cm2lbl[((cm[0]*256+cm[1])*256+cm[2])] = i
    image = np.array(image,dtype="int64")
    ix = ((image[:,:,0]*256+image[:,:,1]*256+image[:,:,2]))
    image2 = cm2lbl[ix]
    return image2 
def rand_crop(data,label,high,width):
    im_width,im_high=data.size
    left=np.random.randint(0,im_width-width)
    top = np.random.randint(0,im_high-high)
    right = left+width
    bottom = top+high
    data = data.crop((left,top,right,bottom))
    label = label.crop((left,top,right,bottom))
    return data,label

def img_transforms(data,label,high,width,colormap):
    data,label = rand_crop(data,label,high,width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    data = data_tfs(data)
    label = torch.from_numpy(image2label(label,colormap))
    return data,label
transform = transforms.Compose([
    transforms.ToTensor()
])

def read_image_path(root="data/ImageSets/Segmentation/train.txt"):
    image = np.loadtxt(root,dtype=str)
    n = len(image)
    data,label = [None]*n , [None]*n
    for i,fname in enumerate(image):
        data[i] = "data/JPEGImages/%s.jpg"%(fname)
        label[i] = "data/SegmentationClass/%s.png"%(fname)
    return data,label
class MyDataset(Data.Dataset):
    def __init__(self,data_root,high,width,imtransform,colormap):
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform =imtransform
        self.colormap = colormap
        data_list,label_list = read_image_path(root=data_root)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
    def _filter(self,images):
        return [im for im in images if (Image.open(im).size[1]>high and Image.open(im).size[0]>width)]
    def __getitem__(self,idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img,label = self.imtransform(img,label,self.high,self.width,self.colormap)
        return img,label
    def __len__(self):
        return len(self.data_list)
high,width = 320,480
voc_train = MyDataset('D:/archive/VOC2012/ImageSets/Segmentation/train.txt',high,width,img_transforms,colormap)
voc_val = MyDataset('D:/archive/VOC2012/ImageSets/Segmentation/val.txt',high,width,img_transforms,colormap)
train_loader = Data.DataLoader(voc_train,batch_size=4,shuffle=True,num_workers=0,pin_memory=False)
val_loader = Data.DataLoader(voc_val,batch_size=4,shuffle=True,num_workers=0,pin_memory=False)
for step,(b_x,b_y) in enumerate(train_loader):
    if step >0:
        break
print('b_x.shape:',b_x.shape)
print('b_y.shape:',b_y.shape)

def inv_normalize_image(data):
    rgb_mean = np.array([0.485,0.456,0.406])
    rgb_std = np.array([0.229,0.224,0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0,1)
def label2image(prelabel,colormap):
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel ==ii)
        image[index,:] = colormap[ii]
    return image.reshape(h,w,3)
b_x_numpy = b_x.data.numpy()
b_x_numpy = b_x_numpy.transpose(0,2,3,1)
b_y_numpy = b_y.data.numpy()
plt.figure(figsize=(16,6))
for ii in range(4):
    plt.subplot(2,4,ii+1)
    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(2,4,ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()

# if __name__ == '__main__':
#     from torch.nn.functional import one_hot
#     data = MyDataset('data')
#     print(data[0][0].shape)
#     print(data[0][1].shape)
#     out=one_hot(data[0][1].long())
#     print(out.shape)
