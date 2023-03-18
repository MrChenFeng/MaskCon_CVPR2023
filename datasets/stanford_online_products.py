import json
import os

from PIL import Image
from torch.utils.data import Dataset


class StanfordOnlineProducts(Dataset):
    """
            SOP-Split1
            # 2517 train classes[25,368 images], 2518 test classes[25,278 images], no overlapping
            # At least 8 images each class, 8~12 images
            SOP-Split2
            # 1498 train classes[14,980 images], 1498 test classes[2,996 images], same class
            # 12 images each class
    """

    def __init__(self, root, train=True, transform=None, split='1'):
        super(StanfordOnlineProducts, self).__init__()
        self.root = root
        self.transform = transform
        if split == '1':
            if train:
                self.image_path = json.load(open(os.path.join(self.root, 'sop_split1/train_path.json'), 'r'))
                self.coarse_labels = json.load(open(os.path.join(self.root, 'sop_split1/train_coarse_label.json'), 'r'))
                self.fine_labels = json.load(open(os.path.join(self.root, 'sop_split1/train_fine_label.json'), 'r'))
            else:
                self.image_path = json.load(open(os.path.join(self.root, 'sop_split1/test_path.json'), 'r'))
                self.coarse_labels = json.load(open(os.path.join(self.root, 'sop_split1/test_coarse_label.json'), 'r'))
                self.fine_labels = json.load(open(os.path.join(self.root, 'sop_split1/test_fine_label.json'), 'r'))
        else: 

            if train:
                self.image_path = json.load(open(os.path.join(self.root, 'sop_split2/train_path.json'), 'r'))
                self.coarse_labels = json.load(open(os.path.join(self.root, 'sop_split2/train_coarse_label.json'), 'r'))
                self.fine_labels = json.load(open(os.path.join(self.root, 'sop_split2/train_fine_label.json'), 'r'))
            else:
                self.image_path = json.load(open(os.path.join(self.root, 'sop_split2/test_path.json'), 'r'))
                self.coarse_labels = json.load(open(os.path.join(self.root, 'sop_split2/test_coarse_label.json'), 'r'))
                self.fine_labels = json.load(open(os.path.join(self.root, 'sop_split2/test_fine_label.json'), 'r'))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path, coarse_label, fine_label = self.image_path[index], self.coarse_labels[index], self.fine_labels[index]
        pil_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, coarse_label, fine_label
