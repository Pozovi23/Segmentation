import csv

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from torch import nn
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

    train(model, train_loader, val_loader, "weights/weight_6", 100)


def test():
    model = Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    ).cuda()

    model.load_state_dict(
        torch.load("/home/gleb/learning/Segmentation/weights/weight_6_87_epochs.pth")
    )

    test_files = get_files("/home/gleb/learning/Segmentation/segmentation_test_set.csv")
    val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
    road_tp = 0
    road_fp = 0
    road_fn = 0

    bg_tp = 0
    bg_fp = 0
    bg_fn = 0
    random.shuffle(test_files)
    for file in test_files:
        print(file)
        result, full_mask = inference(
            model,
            "/home/gleb/learning/Segmentation/dataset/src/pictures_png/" + file[0],
            "/home/gleb/learning/Segmentation/segmentations/" + file[0],
            write=False,
            label_path="/home/gleb/learning/Segmentation/dataset/src/labels_png/" + file[0],
            draw_orig=True
        )
        label_read = cv2.imread(
            "/home/gleb/learning/Segmentation/dataset/src/labels_png/" + file[0]
        )
        label_read = cv2.cvtColor(label_read, cv2.COLOR_BGR2RGB)
        binary_mask = np.all(label_read == [255, 255, 255], axis=-1)
        label = torch.from_numpy(binary_mask).long().cuda()
        pred = torch.from_numpy(full_mask == 255).cuda()

        # просмотр масок
        # plt.subplot(1, 2, 1)
        # plt.imshow(binary_mask)
        # plt.subplot(1, 2, 2)
        # plt.imshow(full_mask)
        # plt.show(block=False)
        # plt.waitforbuttonpress()
        # plt.close()
        # pred = pred.sum(dim = 0)
        # label = label.sum(dim=0)

        pred_road = pred.float()
        road_tp += (pred_road * label).sum().item()
        road_fp += (pred_road * (1 - label)).sum().item()
        road_fn += ((1 - pred_road) * label).sum().item()

        pred_bg = torch.logical_not(pred).float()
        bg_tp += (pred_bg * (1 - label)).sum().item()
        bg_fp += (pred_bg * label).sum().item()
        bg_fn += ((1 - pred_bg) * (1 - label)).sum().item()

        batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
            pred, label, mode="binary", threshold=0.5
        )
        # 1500x1500
        val_tp += batch_tp
        val_fp += batch_fp
        val_fn += batch_fn
        val_tn += batch_tn

        # просмотр фото с наложенными масками
        # plt.figure(figsize=(12, 6))
        # plt.imshow(result)
        # plt.title("Segmentation")
        # plt.show(block=False)
        # plt.waitforbuttonpress()
        # plt.close()

    val_tp = val_tp.unsqueeze(0)
    val_fp = val_fp.unsqueeze(0)
    val_fn = val_fn.unsqueeze(0)
    val_tn = val_tn.unsqueeze(0)
    val_iou = smp.metrics.iou_score(val_tp, val_fp, val_fn, val_tn, reduction="micro")
    val_f1 = smp.metrics.f1_score(val_tp, val_fp, val_fn, val_tn, reduction="micro")

    road_iou = road_tp / (road_tp + road_fp + road_fn + 1e-6)

    bg_iou = bg_tp / (bg_tp + bg_fp + bg_fn + 1e-6)

    macro_iou = (road_iou + bg_iou) / 2
    print(f"road_iou:{road_iou}")
    print(f"bg_iou:{bg_iou}")
    print(f"macro IoU: {macro_iou}")
    print(f"micro IoU: {val_iou}")
    print(f"f1-score: {val_f1}")
    print(
        f"true micro IoU {(road_tp + bg_tp) / (road_tp + road_fp + road_fn + bg_tp + bg_fp + bg_fn + 1e-6)}"
    )


def save():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    ).cuda()

    model.load_state_dict(
        torch.load("/home/gleb/learning/Segmentation/weights/weight_3_34_epochs.pth")
    )

    model.train()

    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    exit()

    model.eval()

    input = torch.ones((1, 3, 224, 224)).cuda()

    onnx_filename = "unet_resnet34_2.onnx"

    torch.onnx.export(
        model,
        input,
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(model)


test()
# save()
# main()
