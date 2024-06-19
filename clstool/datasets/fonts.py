import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision.utils import save_image
from typing import Any, Callable, List, Optional, Union, Tuple
import torch


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
        transform: Optional[Callable] = None,
        attr_label: dict = None,
        resume_size=0.2, 
        n_letter=52, n_size=3, n_fcolor=10, n_bcolor=10, n_font=10,
    ) -> None:
        super(Fonts_imgfolder, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        if isinstance(self.transform, dict):
            self.transform = self.transform[self.split]
        
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
        
        self.attr_label = attr_label

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
            
        if self.attr_label is not None:
            pass  # 获得main_attr属性的索引序号
            for i in range(len(self.attr_label['sub_attr'])):
                pass  # 获得sub_attr属性的索引序号
        
        main_attr = self.Letters.index(letter)
        sub_attrs = [self.Sizes.index(size), self.Colors.index(font_color), self.Colors.index(back_color), self.All_fonts.index(font)]
        # sub_attrs_n = [size, font_color, back_color, font]
        
        return img, main_attr, sub_attrs, img_path

    def __len__(self):
        return self.len
    
    def collate_fn(self, batch):
        batch_image, batch_target, batch_sub_targets, batch_filenames = list(zip(*batch))
        batch_image, batch_target, batch_sub_targets = torch.stack(batch_image), torch.tensor(batch_target), torch.tensor(batch_sub_targets)

        label = batch_target.squeeze()
        sub_labels = batch_sub_targets.squeeze()

        return batch_image, label, sub_labels, batch_filenames