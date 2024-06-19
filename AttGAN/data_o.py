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
        attr_label: list = ["Letter", "Size", "Font_color", "Back_color", "Font"],
        resume_size=0.2, 
        n_letter=52, n_size=3, n_fcolor=5, n_bcolor=5, n_font=50,
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
        self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0), 'cyan': (0, 255, 255), 'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30), 'silver': (192, 192, 192)}
        self.Colors = list(self.Colors.keys())  # color 10
        self.Font_Colors = self.Colors[0:5]  # font_color 5
        self.Back_Colors = self.Colors[5:10]  # backcolor 5
        
        self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
        self.Sizes = list(self.Sizes.keys())  # size 3
        
        for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
            cates = dirs
            break
        self.All_fonts = cates  # style 100
        # print('cates: ', cates)
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
        
        
        if self.split=='train':
            print('train_fonts: ', self.train_fonts)
            self.split_fonts = self.train_fonts
            self.n_fonts = len(self.train_fonts)
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * self.n_fonts
        elif self.split=='valid':
            print('valid_fonts: ', self.valid_fonts)
            self.split_fonts = self.valid_fonts
            self.n_fonts = len(self.valid_fonts)
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * self.n_fonts
        elif self.split=='test':
            print('test_fonts: ', self.test_fonts)
            self.split_fonts = self.test_fonts
            self.n_fonts = len(self.test_fonts)
            self.len = self.n_letter * self.n_size * self.n_fcolor * self.n_bcolor * self.n_fonts
        else:
            raise ValueError("wrong split")
        
        self.attr_label = attr_label

        # self.img_path_lists = []
        # self.img_label_lists = []
        # for m in range(len(self.Letters)):
        #     for n in range(len(self.Sizes)):
        #         for x in range(len(self.Font_Colors)):
        #             for y in range(len(self.Back_Colors)):
        #                 for z in range(len(self.split_fonts)):

        #                     letter = self.Letters[m]
        #                     size = self.Sizes[n]
        #                     font_color = self.Font_Colors[x]
        #                     back_color = self.Back_Colors[y]
        #                     font = self.split_fonts[z]
        #                     label_list = [letter, size, font_color, back_color, font]

        #                     dir_str = '/'.join(label_list)
        #                     name_str = '_'.join(label_list)
        #                     path_str = ''.join([self.root, '/', dir_str, '/', name_str, '.png'])

        #                     self.img_path_lists.append(path_str)
        #                     self.img_label_lists.append(label_list)

    def find_label_m(self, index, split='train'):
        # random choose a C image
        C_letter  = self.Letters[index % self.n_letter]
        index = index // self.n_letter
        
        C_size = self.Sizes[index % self.n_size]
        index = index // self.n_size
        
        C_font_color = self.Font_Colors[index % self.n_fcolor]
        index = index // self.n_fcolor

        C_back_color = self.Back_Colors[index % self.n_bcolor]
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
        
        attrs = [self.Letters.index(C_letter) / self.n_letter, 
                 self.Sizes.index(C_size) / self.n_size, 
                 self.Colors.index(C_font_color) / self.n_fcolor, 
                 self.Colors.index(C_back_color) / self.n_bcolor, 
                 self.All_fonts.index(C_font) / self.n_fonts]

        # attrs_n = [letter, size, font_color, back_color, font]
        
        return C_img_path, attrs

    def find_label_b(self, index, split='train'):
        # random choose a C image
        B_letter = [0]
        I_letter = index % self.n_letter
        Letter  = self.Letters[I_letter]  # task attr
        if I_letter <= 25:
            B_letter[0] = 1
        index = index // self.n_letter
        
        B_size = [0] * 3
        I_size = index % self.n_size
        Size = self.Sizes[I_size]  # sensitive attr1
        B_size[I_size] = 1
        index = index // self.n_size
        
        B_font_color = [0] * 5
        I_font_color = index % self.n_fcolor
        Font_Color = self.Font_Colors[I_font_color]  # sensitive attr2
        B_font_color[I_font_color] = 1
        index = index // self.n_fcolor

        B_back_color = [0] * 5
        I_back_color = index % self.n_bcolor
        Back_Color = self.Back_Colors[I_back_color]  # sensitive attr3
        B_back_color[I_back_color] = 1
        index = index // self.n_bcolor

        if split=='train':
            Font = self.train_fonts[index % len(self.train_fonts)]    # other attr
        elif split=='valid':
            Font = self.valid_fonts[index % len(self.valid_fonts)]
        elif split=='test':
            Font = self.test_fonts[index % len(self.test_fonts)]
        else:
            raise ValueError("wrong split")
        
        C_img_name = Letter + '_' + Size + '_' + Font_Color + '_' + Back_Color + '_' + Font + ".png"
        C_img_path = os.path.join(self.root, Letter, Size, Font_Color, Back_Color, Font, C_img_name)
        
        attrs = B_letter + B_size + B_font_color + B_back_color
        attrs_n = [Letter, Size, Font_Color, Back_Color, Font]
        
        return C_img_path, attrs

    def find_label_l(self, index):
        img_path = self.img_path_lists[index]
        img_label = self.img_label_lists[index]

        B_letter = [0]
        I_letter = self.Letters.index(img_label[0])
        if I_letter <= 25:
            B_letter[0] = 1

        B_size = [0] * 3
        I_size = self.Sizes.index(img_label[1])
        B_size[I_size] = 1

        B_font_color = [0] * 5
        I_font_color = self.Font_Colors.index(img_label[2])
        B_font_color[I_font_color] = 1

        B_back_color = [0] * 5
        I_back_color = self.Back_Colors.index(img_label[3])
        B_back_color[I_back_color] = 1

        attrs = B_letter + B_size + B_font_color + B_back_color
        return img_path, attrs

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''

        # img_path, attrs= self.find_label_m(index, split=self.split)
        img_path, attrs= self.find_label_b(index, split=self.split)
        # img_path, attrs= self.find_label_l(index)
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

    fonts_attrs = ['capital', 'small_size', 'medium_size', 'large_size', 
                   'red_font', 'orange_font', 'yellow_font', 'green_font', 'cyan_font', 
                   'blue_back', 'purple_back', 'pink_back', 'chocolate_back', 'silver_back']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=fonts_attrs, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../data/fonts-v1')  # ../data/celeba/img_align_celeba
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='../data/celeba/list_attr_celeba.txt')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ', 'fonts-v1'], default='fonts-v1')
    args = parser.parse_args()

    print(f'Loading data from {args.data_path}')
    if args.data == 'CelebA':
        from data import CelebA
        train_dataset = CelebA(args.data_path, args.attr_path, 128, 'train', args.attrs)
        valid_dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, drop_last=True
        )
        valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    elif args.data == 'fonts-v1':
        train_dataset = Fonts_imgfolder(args.data_path, 128, 'train', args.attrs)
        valid_dataset = Fonts_imgfolder(args.data_path, 128, 'valid', args.attrs)
        train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn
        )
        valid_dataloader = data.DataLoader(
            valid_dataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=valid_dataset.collate_fn
        )
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