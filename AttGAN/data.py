# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from typing import Any, Callable, List, Optional, Union, Tuple


class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int_)
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    
    def __len__(self):
        return len(self.images)

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int_)
        
        if mode == 'train':
            self.images = images[:162770]
            self.labels = labels[:162770]
        if mode == 'valid':
            self.images = images[162770:182637]
            self.labels = labels[162770:182637]
        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length

class CelebA_HQ(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int_)
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int_)
        
        images = ['{:d}.jpg'.format(i) for i in range(30000)]
        labels = orig_labels[indices]
        
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length

def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch

class Fonts_imgfolder(data.Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''
    def __init__(
        self, 
        root: str,
        image_size:int = 224,
        split: str = "train",
        selected_attrs: list = ['letter'],
        resume_size=1/3, 
        n_letter=52, n_size=3, n_fcolor=5, n_bcolor=5, n_font=70,
    ) -> None:
        super(Fonts_imgfolder, self).__init__()
        self.root = root
        self.split = split
        self.transform = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.n_letter, self.n_size, self.n_fcolor, self.n_bcolor, self.n_font = n_letter, n_size, n_fcolor, n_bcolor, n_font
        
        '''refer'''
        self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))] # letter 52

        self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0), 'cyan': (0, 255, 255), 'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30), 'silver': (192, 192, 192)}
        self.Colors = list(self.Colors.keys())  # color 10
        self.Font_Colors = self.Colors[0:5]  # font_color 5
        self.Back_Colors = self.Colors[5:10]  # backcolor 5
        
        self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
        self.Sizes = list(self.Sizes.keys())  # size 3

        print(f"pwd:{os.getcwd()}")
        print(f"==>> os.path.join(self.root, 'A', 'medium', 'red', 'orange'): {os.path.join(self.root, 'A', 'medium', 'red', 'orange')}")
        for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
            print(f"==>> roots: {roots}")
            dirs.sort()
            cates = dirs
            break
        self.All_fonts = cates  # style 100
        self.Fonts = self.All_fonts[:self.n_font]  # style 100
        # print(f"==>> self.Fonts: {self.Fonts}")
        # print(f"==>> self.All_fonts: {self.All_fonts}")
        # print('==>> self.All_fonts: ', len(self.All_fonts))

        self.attr_list = ['letter'] + ['capital'] + self.Sizes + self.Colors + self.Fonts
        # print(f"==>> self.attr_list: {self.attr_list}")

        
        # 划分训练集和剩余数据
        self.resume_size = resume_size
        indices = self.Sizes.copy()
        split_num = int(np.floor(self.resume_size * self.n_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.train_sizes, self.resume_sizes = indices[split_num:], indices[:split_num]
        
        # 划分验证集和测试集
        self.test_size = 0.5
        indices_ = self.resume_sizes.copy()
        split_num_ = int(np.floor(self.test_size * len(indices_)))
        np.random.seed(42)
        np.random.shuffle(indices_)
        self.valid_sizes, self.test_sizes = indices_[split_num_:], indices_[:split_num_]
        
        
        if self.split=='train':
            print('train_sizes: ', self.train_sizes)
            self.split_sizes = self.train_sizes
            self.n_size_s = len(self.train_sizes)
            self.len = self.n_letter * self.n_bcolor * self.n_fcolor * self.n_font * self.n_size_s
        elif self.split=='valid':
            print('valid_sizes: ', self.valid_sizes)
            self.split_sizes = self.valid_sizes
            self.n_size_s = len(self.valid_sizes)
            self.len = self.n_letter * self.n_bcolor * self.n_fcolor * self.n_font * self.n_size_s
        elif self.split=='test':
            print('test_sizes: ', self.test_sizes)
            self.split_sizes = self.test_sizes
            self.n_size_s = len(self.test_sizes)
            self.len = self.n_letter * self.n_bcolor * self.n_fcolor * self.n_font * self.n_size_s
        else:
            raise ValueError("wrong split")
        
        self.selected_attrs = selected_attrs

        self.img_path_lists = []
        self.img_label_lists = []
        for m in range(len(self.Letters)):
            for n in range(len(self.split_sizes)):
                for x in range(len(self.Font_Colors)):
                    for y in range(len(self.Back_Colors)):
                        for z in range(len(self.Fonts)):

                            letter = self.Letters[m]
                            size = self.split_sizes[n]
                            font_color = self.Font_Colors[x]
                            back_color = self.Back_Colors[y]
                            font = self.Fonts[z]
                            label_list = [letter, size, font_color, back_color, font]

                            dir_str = '/'.join(label_list)
                            name_str = '_'.join(label_list)
                            path_str = ''.join([self.root, '/', dir_str, '/', name_str, '.png'])

                            self.img_path_lists.append(path_str)
                            self.img_label_lists.append(label_list)

    def find_label_l(self, index):
        img_path = self.img_path_lists[index]
        img_label = self.img_label_lists[index]

        B_letter = [0]
        B_Capital = [0]
        B_letter[0] = self.Letters.index(img_label[0])
        if B_letter[0] <= 25:
            B_Capital[0] = 1

        B_size = [0] * self.n_size
        I_size = self.Sizes.index(img_label[1])
        B_size[I_size] = 1

        B_font_color = [0] * self.n_fcolor
        I_font_color = self.Font_Colors.index(img_label[2])
        B_font_color[I_font_color] = 1

        B_back_color = [0] * self.n_bcolor
        I_back_color = self.Back_Colors.index(img_label[3])
        B_back_color[I_back_color] = 1

        B_font = [0] * self.n_font
        I_font = self.Fonts.index(img_label[4])
        B_font[I_font] = 1

        attrs_label = B_letter + B_Capital + B_size + B_font_color + B_back_color + B_font
        mapping = dict(zip(self.attr_list, attrs_label))
        attrs_label_select = [mapping[attr] for attr in self.selected_attrs if attr in mapping]

        return img_path, attrs_label_select

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        
        img_path, attrs= self.find_label_l(index)
        # print(f"==>> attrs: {attrs}")
        # print(f"==>> img_path: {img_path}")
        
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, attrs

    def __len__(self):
        return self.len
    
    def collate_fn(self, batch):
        batch_image, batch_targets= list(zip(*batch))
        batch_image, batch_targets = torch.stack(batch_image), torch.tensor(batch_targets)

        # batch_targets = batch_targets.squeeze()

        return batch_image, batch_targets


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    print('Start')

    celeba_attrs = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]

    # font 70
    fonts_attrs = [# 'letter', 
                   'capital', 
                #    'small', 'medium', 'large', 
                   'red', 'orange', 'Yellow', 'green', 'cyan', 
                #    'blue', 'purple', 'pink', 'chocolate', 'silver',
                   'aakar', 'abyssinicasil', 'ani', 'anjalioldlipi', 'arplukaicn', 'arplukaihk', 'arplukaitw', 'arplukaitwmbe', 'arplumingcn', 'arpluminghk', 'arplumingtw', 'arplumingtwmbe', 'chandas', 'chilanka', 'dejavusans', 'dejavusansmono', 'dejavuserif', 'dyuthi', 'freemono', 'freesans', 'freeserif', 'gargi', 'garuda', 'jamrul', 'kalimati', 'karumbi', 'keraleeyam', 'khmeros', 'khmerossystem', 'kinnari', 'liberationmono', 'liberationsans', 'liberationsansnarrow', 'liberationserif', 'likhan', 'loma', 'manjari', 'meera', 'mitramono', 'muktinarrow', 'nakula', 'norasi', 'notomono', 'notosanscjkhk', 'notosanscjkhkblack', 'notosanscjkhkdemilight', 'notosanscjkhklight', 'notosanscjkhkmedium', 'notosanscjkhkthin', 'notosanscjkjp', 'notosanscjkjpblack', 'notosanscjkjpdemilight', 'notosanscjkjplight', 'notosanscjkjpmedium', 'notosanscjkjpthin', 'notosanscjkkr', 'notosanscjkkrblack', 'notosanscjkkrdemilight', 'notosanscjkkrlight', 'notosanscjkkrmedium', 'notosanscjkkrthin', 'notosanscjksc', 'notosanscjkscblack', 'notosanscjkscdemilight', 'notosanscjksclight', 'notosanscjkscmedium', 'notosanscjkscthin', 'notosanscjktc', 'notosansmonocjkhk', 'notosansmonocjkjp'
                   ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ', 'fonts-v1'], default='fonts-v1')
    parser.add_argument('--attrs', dest='attrs', default=celeba_attrs, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../data/celeba/img_align_celeba')  # ../data/celeba/img_align_celeba  ../data/fonts-v1
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='../data/celeba/list_attr_celeba.txt')
    args = parser.parse_args()

    if args.data=='CelebA':
        args.attrs = celeba_attrs
        args.data_path = '../data/celeba/img_align_celeba'
    elif args.data=='fonts-v1':
        args.attrs = fonts_attrs
        args.data_path = '../data/fonts-v1'

    print(f'Loading data from {args.data_path}')
    if args.data == 'CelebA':
        from data import CelebA
        train_dataset = CelebA(args.data_path, args.attr_path, 128, 'train', args.attrs)
        valid_dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
        test_dataset = CelebA(args.data_path, args.attr_path, 128, 'test', args.attrs)
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, drop_last=True
        )
        valid_dataloader = data.DataLoader(
            valid_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        test_dataloader = data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
    elif args.data == 'fonts-v1':
        train_dataset = Fonts_imgfolder(args.data_path, 128, 'train', args.attrs)
        valid_dataset = Fonts_imgfolder(args.data_path, 128, 'valid', args.attrs)
        # test_dataset = Fonts_imgfolder(args.data_path, 128, 'test', args.attrs)
        train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn
        )
        valid_dataloader = data.DataLoader(
            valid_dataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=valid_dataset.collate_fn
        )
        # test_dataloader = data.DataLoader(
        #     test_dataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=valid_dataset.collate_fn
        # )
    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))
    

    print('Attributes:')
    print(args.attrs)
    # for x, y in fonts_attrs:
    #     vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
    #     print(y)
    #     break
    # del x, y
    # imgs, attrs = next(iter(valid_dataloader))
    # print(f"==>> attrs_n: {attrs}")
    # import torchvision.utils as vutils
    # vutils.save_image(imgs, 'output.png', nrow=8, normalize=True)
    for img_a, att_a in valid_dataloader:
        # print(f"==>> img_a: {img_a}")
        pass