import os
import pickle
from typing import Tuple, Optional, Callable, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


def _cifar10_to_cifartoy(split: str, target: int) -> int:
    bad_dict = \
        {0: 0,
         1: 1,
         2: 2,
         3: 3,
         8: 4,
         9: 5,
         5: 6,
         7: 7,
         6: 8,
         4: 9}  # bad splitting
    good_dict = \
        {0: 0,
         1: 1,
         8: 2,
         9: 3,
         2: 4,
         3: 5,
         5: 6,
         7: 7,
         6: 8,
         4: 9}  # good splitting
    if split == 'good':
        return good_dict[target]
    else:
        return bad_dict[target]


class CIFARtoy(Dataset):
    """`Artificial toy CIFAR10 Dataset. Modified based on original pytorch dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

        """
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            split: str,  # good or bad
            train: bool = True,
            transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        if split == 'good':
            all = [0, 1, 8, 9, 2, 3, 5, 7]  # good splitting
        else:
            all = [0, 1, 5, 3, 8, 9, 2, 7]  # bad splitting
        self.class_1 = all[:4]
        self.class_2 = all[4:]

        super(CIFARtoy, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # train or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        # get interested sample out
        targets = np.array(self.targets)
        coarse_targets = targets.copy()
        selected_id1 = np.array([], dtype=int)
        for i in self.class_1:
            cur_id = np.where(targets == i)[0]
            selected_id1 = np.concatenate([selected_id1, cur_id])
        selected_id2 = np.array([], dtype=int)
        for i in self.class_2:
            cur_id = np.where(targets == i)[0]
            selected_id2 = np.concatenate([selected_id2, cur_id])
        # print(selected_id1)
        # re-sort the indexes
        coarse_targets[selected_id1] = 0
        coarse_targets[selected_id2] = 1
        selected_id = np.concatenate([selected_id1, selected_id2])

        new_ = targets.copy()
        for idx, target in enumerate(targets):
            new_[idx] = _cifar10_to_cifartoy(split, target)
        targets = new_

        # select only interested indexes
        self.data = self.data[selected_id]
        self.targets = targets[selected_id]
        self.coarse_targets = coarse_targets[selected_id]

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, fine_target, coarse_target = self.data[index], self.targets[index], self.coarse_targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, coarse_target, fine_target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
