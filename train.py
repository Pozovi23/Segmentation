import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, val_loader, path_to_save_model, epochs):
    writer = SummaryWriter("runs/Segmentation_6")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Number of epoch: {epoch}")
        total_train_loss = 0
        num_of_train_batches = 0
        model.train()
        for image, label in train_loader:
            image = image.cuda()
            label = label.cuda().unsqueeze(1)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            total_train_loss += loss.item()
            num_of_train_batches += 1
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", total_train_loss / num_of_train_batches, epoch)

        model.eval()

        total_validation_loss = 0
        num_of_validation_batches = 0

        road_tp = 0
        road_fp = 0
        road_fn = 0

        bg_tp = 0
        bg_fp = 0
        bg_fn = 0

        for image, label in val_loader:
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda().unsqueeze(1)
                output = model(image)
                loss = criterion(output, label)
                pred = torch.sigmoid(output) > 0.5

                pred_road = pred.float()
                road_tp += (pred_road * label).sum().item()
                road_fp += (pred_road * (1 - label)).sum().item()
                road_fn += ((1 - pred_road) * label).sum().item()

                pred_bg = torch.logical_not(pred).float()
                bg_tp += (pred_bg * (1 - label)).sum().item()
                bg_fp += (pred_bg * label).sum().item()
                bg_fn += ((1 - pred_bg) * (1 - label)).sum().item()

                total_validation_loss += loss.item()
                num_of_validation_batches += 1

        road_iou = road_tp / (road_tp + road_fp + road_fn + 1e-9)
        bg_iou = bg_tp / (bg_tp + bg_fp + bg_fn + 1e-9)
        macro_iou = (road_iou + bg_iou) / 2
        micro_iou = (road_tp + bg_tp) / (
            road_tp + road_fp + road_fn + bg_tp + bg_fp + bg_fn + 1e-6
        )
        writer.add_scalar(
            "Loss/validation", total_validation_loss / num_of_validation_batches, epoch
        )

        writer.add_scalar("IoU/Road", road_iou, epoch)

        writer.add_scalar("IoU/Background", bg_iou, epoch)

        writer.add_scalar("IoU/Macro", macro_iou, epoch)

        writer.add_scalar("IoU/Micro", micro_iou, epoch)

        torch.save(model.state_dict(), path_to_save_model + f"_{epoch}_epochs.pth")

    writer.close()
    # torch.save(model.state_dict(), path_to_save_model + "_last.pth")
