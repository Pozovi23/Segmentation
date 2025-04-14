import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from inference import inference
from train import train


def get_files(path):
    files = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            files.append(line)

    return files


def main():
    model = Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    ).cuda()

    train_files = get_files(
        "/home/gleb/learning/Segmentation/segmentation_train_set.csv"
    )
    validation_files = get_files(
        "/home/gleb/learning/Segmentation/segmentation_validation_set.csv"
    )

    train_dataset = SegmentationDataset(train_files, state="train")
    val_dataset = SegmentationDataset(validation_files, state="val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train(model, train_loader, val_loader, "weights/weight_3", 50)


def test():
    model = Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    ).cuda()

    model.load_state_dict(
        torch.load("/home/gleb/learning/Segmentation/weights/weight_3_34_epochs.pth")
    )

    test_files = get_files("/home/gleb/learning/Segmentation/segmentation_test_set.csv")
    val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
    for file in test_files:
        print(file)
        result, full_mask = inference(
            model,
            "/home/gleb/learning/Segmentation/dataset/src/pictures_png/" + file[0],
            "/home/gleb/learning/Segmentation/segmentations/" + file[0],
            write=True,
        )
        label_read = cv2.imread(
            "/home/gleb/learning/Segmentation/dataset/src/labels_png/" + file[0]
        )
        label_read = cv2.cvtColor(label_read, cv2.COLOR_BGR2RGB)
        binary_mask = np.all(label_read == [255, 255, 255], axis=-1)
        label = torch.from_numpy(binary_mask).long().cuda()
        pred = torch.from_numpy(full_mask == 255).float().cuda()

        # просмотр масок
        # plt.subplot(1, 2, 1)
        # plt.imshow(binary_mask)
        # plt.subplot(1, 2, 2)
        # plt.imshow(full_mask)
        # plt.show(block=False)
        # plt.waitforbuttonpress()
        # plt.close()

        batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
            pred, label, mode="binary", threshold=0.5
        )

        val_tp += batch_tp.sum(dim=0)
        val_fp += batch_fp.sum(dim=0)
        val_fn += batch_fn.sum(dim=0)
        val_tn += batch_tn.sum(dim=0)

        # просмотр фото с наложенными масками
        plt.figure(figsize=(12, 6))
        plt.imshow(result)
        plt.title("Segmentation")
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

    val_tp = val_tp.unsqueeze(0)
    val_fp = val_fp.unsqueeze(0)
    val_tn = val_tn.unsqueeze(0)
    val_fn = val_fn.unsqueeze(0)
    val_iou = smp.metrics.iou_score(val_tp, val_fp, val_fn, val_tn, reduction="micro")
    val_f1 = smp.metrics.f1_score(val_tp, val_fp, val_fn, val_tn, reduction="micro")
    val_accuracy = smp.metrics.accuracy(
        val_tp, val_fp, val_fn, val_tn, reduction="micro"
    )
    val_recall = smp.metrics.recall(val_tp, val_fp, val_fn, val_tn, reduction="micro")

    print(f"IoU: {val_iou}")
    print(f"f1-score: {val_f1}")
    print(f"Accuracy: {val_accuracy}")
    print(f"Recall: {val_recall}")


test()

# main()
