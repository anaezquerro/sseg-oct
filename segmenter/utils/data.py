from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as t
import os, random, cv2, glob, torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Iterable

def flatten(list_of_lists, levels=None):
    items = list()
    for l in list_of_lists:
        if (isinstance(l, Iterable)) and (levels is None or levels != 0):
            items += flatten(l, levels if levels is None else levels-1)
        else:
            items.append(l)
    return items

class OCTDataset(Dataset):
    add_folder = lambda folder, files: [f'{folder}/{file}' for file in files]

    def __init__(self, img_files: List[str], mask_files: List[str], rsize: Tuple[int] =(416, 624),
                 transform: Optional[t.Compose] = None):
        super().__init__()
        self.img_files = img_files
        self.mask_files = mask_files
        self.rsize = rsize


        if transform:
            self.transform = t.Compose([
                t.ToPILImage(),
                transform,
                t.Resize(self.rsize, interpolation=InterpolationMode.NEAREST),
                t.ToTensor()
            ])
        else:
            self.transform = t.Compose([
                t.ToPILImage(),
                t.Resize(self.rsize, interpolation=InterpolationMode.NEAREST),
                t.ToTensor()]
            )


    # Returns both the image and the mask
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)
        mask = plt.imread(mask_path)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if len(image.shape) > 2:
            image = image[:, :, 0]
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)  # Make sure that mask is binary

        # Apply the defined transformations to both image and mask
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply tsshhis seed to image transforms
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)  # apply the same seed to mask transforms
        torch.manual_seed(seed)
        mask = (self.transform(mask) > 100/255)*1.0
        return image, mask

    @classmethod
    def build(cls, image_path: str, mask_path: str, **kwargs):

        # Load all the filenames with extension tif from the image_path directory
        img_files = glob.glob(os.path.join(image_path, '*.jpg'))
        mask_files = []

        # We asume that each image has the same filename as its corresponding mask
        # but it is stored in another directory (mask_path)
        for img_path in img_files:
            mask_files.append(os.path.join(mask_path, os.path.basename(img_path)))

        return OCTDataset(img_files, mask_files, **kwargs)

    @classmethod
    def split(cls, pval: float, image_path: str, mask_path: str, **kwargs):
        base_files = np.array(os.listdir(image_path))
        add_folder = OCTDataset.add_folder

        # split in train and val
        indices = np.random.choice(len(base_files), size=int(pval*len(base_files)), replace=False).tolist()
        val_files = base_files[indices].tolist()
        train_files = np.delete(base_files, indices).tolist()

        transform = kwargs.pop('transform') if 'transform' in kwargs else None

        train = OCTDataset(add_folder(image_path, train_files), add_folder(mask_path, train_files), transform=transform, **kwargs)
        val = OCTDataset(add_folder(image_path, val_files), add_folder(mask_path, val_files), **kwargs)
        return train, val

    @classmethod
    def kfold(cls, k: int, image_path: str, mask_path: str, **kwargs):
        base_files = np.array(os.listdir(image_path))
        add_folder = OCTDataset.add_folder
        transform = kwargs.pop('transform') if 'transform' in kwargs else None

        # shuffle image paths
        np.random.shuffle(base_files)

        # divide in k folds image paths
        m = len(base_files)//k
        base_files = [base_files[i:(i+m)] for i in range(0, len(base_files), m)]

        for i in range(k):
            train_files = base_files.copy()
            val_files = train_files.pop(i)
            train_files = flatten(train_files, levels=1)

            train = OCTDataset(add_folder(image_path, train_files), add_folder(mask_path, train_files), transform=transform, **kwargs)

            positives, negatives, total = 0, 0, 0
            for i in range(len(train)):
                _, label = train[i]
                label = label.flatten()
                total += len(label)
                positives += label.sum().item()
                negatives += (len(label) - label.sum().item())

            val = OCTDataset(add_folder(image_path, val_files), add_folder(mask_path, val_files), **kwargs)

            for i in range(len(val)):
                _, label = val[i]
                total += len(label)
                label = label.flatten()
                positives += label.sum().item()
                negatives += (len(label) - label.sum().item())

            yield train, val

    def __len__(self):
        return len(self.img_files)



