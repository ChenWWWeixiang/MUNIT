"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import SimpleITK as sitk
import torchvision.utils as vutils
def default_loader(path):
    return Image.open(path).convert('RGB')

def mha_loader(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
from skimage.transform import resize
from PIL import Image
import os,random,torch
import os.path
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','mha'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader,sizes=None):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.zl,self.wl,self.hl,self.newl,self.newz=sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        valid_range=np.where(img>10)
        valid_range=[[min(xx),max(xx)] for xx in valid_range]
        if self.transform is not None:
            img=self._randomcrop(img,valid_range)
            img = resize(img, ( self.newz,self.newl, self.newl), order=1, mode='edge',
                                 cval=0, clip=True, preserve_range=True)
            img=img[np.newaxis,:,:,:]
            img[img<-0]=0
            img[img>2000]=2000
            img=(img)*1.0/2000
            img = torch.Tensor(img)
            #self.__write_images([img],1, '/home/data1/MUNIT/tmptest.jpg')

        if self.return_paths:
            return img, path
        else:
            return img

    def __write_images(self,image_outputs, display_image_num, file_name):
        image_outputs = [images[:, 16, :, :].expand( 3, -1, -1) for images in
                         image_outputs]  # expand gray-scale images to 3 channels
        image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
        image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
        vutils.save_image(image_grid, file_name, nrow=1)
    def __len__(self):
        return len(self.imgs)
    def _randomcrop(self,imgs,valid_range):
        sizez=valid_range[0][1]-valid_range[0][0]
        sizew = valid_range[1][1] - valid_range[1][0]
        sizeh = valid_range[2][1] - valid_range[2][0]
        if sizez>self.zl:
            z=random.randint(valid_range[0][0],valid_range[0][1]-self.zl)
        else:
            z=valid_range[0][0]
        if sizew>self.wl:
            x=random.randint(valid_range[1][0],valid_range[1][1]-self.wl)
        else:
            x=valid_range[1][0]
        if sizeh > self.hl:
            y=random.randint(valid_range[2][0],valid_range[2][1]-self.hl)
        else:
            y=valid_range[2][0]
        cropimg=imgs[z:z+self.zl,x:x+self.wl,y:y+self.hl]
        now_size=cropimg.shape
        padding=np.array((self.zl,self.wl,self.hl))-np.array(now_size)
        cropimg=np.pad(cropimg,((padding[0]//2,padding[0]-padding[0]//2),
                                (padding[1]//2,padding[1]-padding[1]//2),
                                (padding[2]//2,padding[2]-padding[2]//2)),mode='reflect')
        return cropimg
