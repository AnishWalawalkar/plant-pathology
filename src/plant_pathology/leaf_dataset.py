import torch
import torch.utils.data as Data
import numpy as np
from torchvision import transforms as T
from PIL import Image

from src.plant_pathology.image_utils import IMAGENET_MEAN, IMAGENET_STD


class LeafDataset(Data.Dataset):
    def __init__(self, img_paths, labels=None, train=True, test=False):
        self.img_paths = img_paths
        self.train = train
        self.test = test

        if not self.test:
            self.labels = labels

        self.train_transform = T.Compose([T.RandomRotation(25),
                                          T.RandomHorizontalFlip(),
                                          T.RandomVerticalFlip(), ])
        self.test_transform = T.Compose([T.RandomRotation(25),
                                         T.RandomHorizontalFlip(),
                                         T.RandomVerticalFlip()])
        self.default_transform = T.Compose([T.ToTensor(),
                                            T.Normalize(IMAGENET_MEAN,
                                                        IMAGENET_STD), ])  # ImageNet Stats

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, i):
        image = Image.open(self.img_paths[i]).resize((512, 512))
        if not self.test:
            label = torch.tensor(np.argmax(self.labels.loc[i, :].values))

        if self.train:
            image = self.train_transform(image)
        elif self.test:
            image = self.test_transform(image)
        image = self.default_transform(image)

        if self.test:
            return image

        return image, label