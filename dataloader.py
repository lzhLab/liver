# -*- coding:utf-8 -*-

"""
@author:gz
@file:dataload.py
@time:2021/2/1115:45
"""

from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
import glob

def LoaderNames(names_path):
    f = open(names_path, 'r')
    names = f.read().splitlines()
    f.close()
    return names

def default_loader(path):
    train_3D = nib.load(path)
    label = train_3D.dataobj[:, :, 0]/255.
    #3 ct slices
    data = train_3D.dataobj[:,:, 3:6]/255.
    return data,label

class MyDataset(Dataset):
    def __init__(self, model_type, data_filename, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = glob.glob(os.path.join(data_filename,model_type+'/*'))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.data_filename = data_filename

    def __getitem__(self, index):
        img_str = self.imgs[index]
        img,label = self.loader(img_str)
        if self.transform is not None:
            img = self.transform(img)  # to Tensor
            label = self.transform(label)  # to Tensor
        return img, label

    def __len__(self):
        return len(self.imgs)

# train_data = MyDataset(model_type='train', data_filename='data_v2', transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=4)
