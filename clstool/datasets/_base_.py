import os.path
from typing import Optional, Callable, Dict, Union

import torch
from torch.utils.data import Dataset

from ..utils.io import pil_loader

__all__ = ["BaseDataset"]


def default_loader(path, format="RGB"):
    return pil_loader(path, format)


class BaseDataset(Dataset):
    """
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split. E.g, ``train``, ``val``, ``test``...
        transform (callable, optional): A function/transform that takes in an PIL image and transforms it.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        batch_transform (callable, optional): A function/transform that takes in a batch and transforms it.
        loader (callable, optional): A function to load an image given its path.
        verbose (bool): Whether to print information.
    """

    def __init__(
            self,
            root: str,
            split: str,
            transform: Union[Optional[Callable], Dict[str, Optional[Callable]]] = None,
            target_transform: Union[Optional[Callable], Dict[str, Optional[Callable]]] = None,
            batch_transform: Union[Optional[Callable], Dict[str, Optional[Callable]]] = None,
            loader: Optional[Callable] = None,
            verbose: bool = True
    ):
        if root is None:
            root = f'./data/{self.__class__.__name__.lower()}'
        if loader is None:
            loader = default_loader
        self.root = os.path.expanduser(root)
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.batch_transform = batch_transform
        self.loader = loader

        self.__check_transforms()

        if verbose:
            print(f'Loading {self.__class__.__name__.lower()}-{split} from {self.root}')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        batch_image, batch_target, batch_sub_targets, batch_filenames = list(zip(*batch))
        batch_image, batch_target, batch_sub_targets = torch.stack(batch_image), torch.tensor(batch_target), torch.tensor(batch_sub_targets)

        label = batch_target.squeeze()
        sub_labels = batch_sub_targets.squeeze()

        if self.batch_transform is not None:
            batch_image, batch_target = self.batch_transform(batch_image, batch_target)

        return batch_image, label, sub_labels, batch_filenames

    def __check_transforms(self):
        if isinstance(self.transform, dict):
            self.transform = self.transform[self.split]
        if isinstance(self.target_transform, dict):
            self.target_transform = self.target_transform[self.split]
        if isinstance(self.batch_transform, dict):
            self.batch_transform = self.batch_transform[self.split]
