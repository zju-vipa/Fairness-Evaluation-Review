"""dataset.py"""

import os
import PIL
import numpy as np
import csv
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg, extract_archive
from typing import Any, Callable, List, Optional, Union, Tuple


CSV = namedtuple("CSV", ["header", "index", "data"])


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    # assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'celeba')
        # transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),])
        train_kwargs = {'root':root, 'image_size':image_size}
        dset = CelebA
        
    elif name.lower() == 'fonts-v1':
        root = os.path.join(dset_dir, 'fonts-v1')
        # transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),])
        train_kwargs = {'root':root, 'image_size':image_size}
        dset = Fonts_imgfolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

class Fonts_imgfolder(Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''
    def __init__(
        self, 
        root: str,
        split: str = "train",
        # transform: Optional[Callable] = None,
        resume_size=0.2, 
        image_size = 64,
        n_letter=52, n_size=3, n_fcolor=10, n_bcolor=10, n_font=10,
    ) -> None:
        super(Fonts_imgfolder, self).__init__()
        self.root = root
        self.split = split
        # self.transform = transform
        # if isinstance(self.transform, dict):
        #     self.transform = self.transform[self.split]
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        self.n_letter, self.n_size, self.n_fcolor, self.n_bcolor, self.n_font = n_letter, n_size, n_fcolor, n_bcolor - 1, n_font
        
        '''refer'''
        self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
                    'cyan': (0, 255, 255),
                    'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
                    'silver': (192, 192, 192)}
        self.Colors = list(self.Colors.keys()) # color 10
        
        self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
        self.Sizes = list(self.Sizes.keys())  # size 3
        
        for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
            cates = dirs
            break
        # print('cates: ', cates)
        self.All_fonts = cates  # style 100
        # print('self.All_fonts: ', len(self.All_fonts))
        
        self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))] # letter 52
        
        # 划分训练集和剩余数据
        self.resume_size = resume_size
        indices = self.All_fonts.copy()[:self.n_font]
        split_num = int(np.floor(self.resume_size * self.n_font))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.train_fonts, self.resume_fonts = indices[split_num:], indices[:split_num]
        
        # 划分验证集和测试集
        self.test_size = 0.5
        indices_ = self.resume_fonts.copy()
        split_num_ = int(np.floor(self.test_size * len(indices_)))
        np.random.seed(42)
        np.random.shuffle(indices_)
        self.valid_fonts, self.test_fonts = indices_[split_num_:], indices_[:split_num_]
        
        print('train_fonts: ', self.train_fonts)
        print('valid_fonts: ', self.valid_fonts)
        print('test_fonts: ', self.test_fonts)
        
        if self.split=='train':
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * len(self.train_fonts)
        elif self.split=='valid':
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * len(self.valid_fonts)
        elif self.split=='test':
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * len(self.test_fonts)
        else:
            raise ValueError("wrong split")

    def findN(self, index, split='train'):
        # random choose a C image
        C_letter  = self.Letters[index % self.n_letter]
        index = index // self.n_letter
        
        C_size = self.Sizes[index % self.n_size]
        index = index // self.n_size
        
        C_font_color = self.Colors[index % self.n_fcolor]
        index = index // self.n_fcolor
        
        resume_colors = self.Colors.copy()
        resume_colors.remove(C_font_color)
        C_back_color = resume_colors[index % self.n_bcolor]
        index = index // self.n_bcolor

        if split=='train':
            C_font = self.train_fonts[index % len(self.train_fonts)]
        elif split=='valid':
            C_font = self.valid_fonts[index % len(self.valid_fonts)]
        elif split=='test':
            C_font = self.test_fonts[index % len(self.test_fonts)]
        else:
            raise ValueError("wrong split")
        
        C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)
        
        return C_img_path, C_letter, C_size, C_font_color, C_back_color, C_font

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''

        img_path, letter, size, font_color, back_color, font= self.findN(index, split=self.split)
        
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return self.len
    
class CelebA(Dataset):
    # base_folder = "celeba"
    def __init__(self, 
        root: str, 
        split: str = "train",
        image_size = 64,  
        attr_label = None, target_type = "attr"):
        super(CelebA, self).__init__()
        self.root = root
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")
        self.split = split
        # images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")  # 获得划分结果 0,1,2
        attr = self._load_csv("list_attr_celeba.txt", header=1)
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()  # 提取出需要处理的划分集
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index  # 如果全要就都给
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]  # 如果只要片段就提取出片段
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header  # 第一行内容作为属性名
        self.attr_label = attr_label
        
        # if split == 'train':
        #     self.images = images[:162770]  # 因为第一行是总数
        # if split == 'valid':
        #     self.images = images[162770:182637]
        # if split == 'test':
        #     self.images = images[182637:]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
                                       
        # self.length = len(self.images)
        self.length = len(self.attr)
    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> csv:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))
    def __getitem__(self, index):
        # img = self.transform(Image.open(os.path.join(self.root, self.images[index])))
        X = PIL.Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))  # 到对应路径取出对应图片

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])  # 获得第index张图片对应的40个属性值
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            
            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            if self.attr_label is not None:
                main_attr_index= self.attr_names.index(self.attr_label['main_attr'])  # 获得main_attr属性的索引序号
                main_attr = target[int(main_attr_index)]  # 获得main_attr属性值
                sub_attr = []
                for i in range(len(self.attr_label['sub_attr'])):
                    sub_attr_index= self.attr_names.index(self.attr_label['sub_attr'][i])  # 获得sub_attr属性的索引序号
                    sub_attr.append(target[int(sub_attr_index)])  # 获得sub_attr的属性值 
        else:
            target = None
            main_attr = None
            sub_attr = None

        return X
        # return img
    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
