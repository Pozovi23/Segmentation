import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class SegmentationDataset(Dataset):
    def __init__(self, files, state):
        self.images_and_labels = files
        if state == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=(-30, 30)),
                    A.Normalize(mean=(0.336, 0.340, 0.297), std=(0.192, 0.183, 0.183)),
                    A.pytorch.ToTensorV2(),
                ],
                additional_targets={"image": "label"},
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.336, 0.340, 0.297), std=(0.192, 0.183, 0.183)),
                    A.pytorch.ToTensorV2(),
                ],
                additional_targets={"image": "label"},
            )

    def __len__(self):
        return len(self.images_and_labels)

    def __getitem__(self, idx):
        path = self.images_and_labels[idx][0]
        image = read_image("/home/gleb/learning/Segmentation/dataset/pictures/" + path)
        label = read_image("/home/gleb/learning/Segmentation/dataset/labels/" + path)
        augmented = self.transform(image=image, label=label)
        binary_mask = np.all(augmented["label"] == [255, 255, 255], axis=-1)
        return augmented["image"], torch.from_numpy(binary_mask).float()
