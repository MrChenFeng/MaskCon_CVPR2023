from typing import Tuple, NoReturn, Optional
import os
import sys
import pickle

from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


def _cifar100_to_cifar20(target: int) -> int:
    _dict = \
        {0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13}

    return _dict[target]


class CIFAR100(Dataset):
    """CIFAR100-20 Dataset.

    Modified based on pytorch `CIFAR100` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root: str = './data/cifar-20',
                 train: bool = True,
                 transform: Optional = None,
                 download: bool = False):

        super(CIFAR100, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # train or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.sub_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.sub_targets.extend(entry['fine_labels'])
                self.targets.extend(entry['coarse_labels'])


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        # Remap classes from cifar-100 to cifar-20
        # new_ = self.targets
        # for idx, target in enumerate(self.targets):
        #     new_[idx] = _cifar100_to_cifar20(target)
        # self.targets = new_
        self.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                        'household electrical devices', 'househould furniture', 'insects', 'large carnivores',
                        'large man-made outdoor things', 'large natural outdoor scenes',
                        'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people',
                        'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

    def _load_meta(self) -> NoReturn:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index
        Returns:
            Tuple: image, class
        """
        img, coarse_target, fine_target = Image.fromarray(self.data[index]), self.targets[index], self.sub_targets[index]
        # print(img)
        if self.transform is not None:
            img = self.transform(img)
        # print(img)
        return img, coarse_target, fine_target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> NoReturn:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
