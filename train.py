import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, val_loader, path_to_save_model, epochs):
    writer = SummaryWriter("runs/Segmentation_2")
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
            writer.add_scalar(
                "Loss/train each batch", loss.item(), num_of_train_batches
            )
            total_train_loss += loss.item()
            num_of_train_batches += 1
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", total_train_loss / num_of_train_batches, epoch)

        model.eval()

        total_validation_loss = 0
        num_of_validation_batches = 0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0

        for image, label in val_loader:
            with torch.no_grad():
                image = image.cuda()
                label = label.cuda().unsqueeze(1)
                output = model(image)
                loss = criterion(output, label)
                preds = torch.sigmoid(output)
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    (preds > 0.5).float(), label.long(), mode="binary", threshold=0.5
                )
                val_tp += batch_tp.sum(dim=0)
                val_fp += batch_fp.sum(dim=0)
                val_fn += batch_fn.sum(dim=0)
                val_tn += batch_tn.sum(dim=0)
                writer.add_scalar(
                    "Loss/validation each batch", loss.item(), num_of_validation_batches
                )
                total_validation_loss += loss.item()
                num_of_validation_batches += 1

        val_tp = val_tp.unsqueeze(0)
        val_fp = val_fp.unsqueeze(0)
        val_tn = val_tn.unsqueeze(0)
        val_fn = val_fn.unsqueeze(0)
        val_iou = smp.metrics.iou_score(
            val_tp, val_fp, val_fn, val_tn, reduction="micro"
        )
        val_f1 = smp.metrics.f1_score(val_tp, val_fp, val_fn, val_tn, reduction="micro")
        val_accuracy = smp.metrics.accuracy(
            val_tp, val_fp, val_fn, val_tn, reduction="micro"
        )
        val_recall = smp.metrics.recall(
            val_tp, val_fp, val_fn, val_tn, reduction="micro"
        )

        writer.add_scalar("Metrics_val/IoU", val_iou, epoch)
        writer.add_scalar("Metrics_val/F1", val_f1, epoch)
        writer.add_scalar("Metrics_val/Accuracy", val_accuracy, epoch)
        writer.add_scalar("Metrics_val/Recall", val_recall, epoch)

        writer.add_scalar(
            "Loss/validation", total_validation_loss / num_of_validation_batches, epoch
        )

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), path_to_save_model + f"_{epoch}_epochs.pth")

    writer.close()
    torch.save(model.state_dict(), path_to_save_model + "_last.pth")
