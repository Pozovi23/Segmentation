import csv

import cv2
import numpy as np
import torch
from torchvision import transforms


def inference(model, picture_path, save_path, write=False):
    img = cv2.imread(picture_path)
    window_size = 224
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.336, 0.340, 0.297), std=(0.192, 0.183, 0.183)),
        ]
    )
    model.eval()
    full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if img.shape[0] >= 224 and img.shape[1] >= 224:
        for y in range(0, img.shape[0] - window_size + 1, window_size):
            for x in range(0, img.shape[1] - window_size + 1, window_size):
                crop = img_rgb[y : y + window_size, x : x + window_size]
                crop_tensor = transform(crop).unsqueeze(0).cuda()
                with torch.no_grad():
                    crop_output = model(crop_tensor)
                    pred_mask = torch.sigmoid(crop_output).squeeze().cpu().numpy()
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                full_mask[y : y + window_size, x : x + window_size] += pred_mask

            if img.shape[1] % window_size != 0:
                crop = img_rgb[
                    y : y + window_size,
                    (img.shape[1] // window_size) * window_size : img.shape[1],
                ]
                helper = np.zeros((224, 224, 3), dtype=np.uint8)
                helper[
                    0:224,
                    0 : (img.shape[1] - (img.shape[1] // window_size) * window_size),
                    :,
                ] += crop
                crop_tensor = transform(helper).unsqueeze(0).cuda()
                with torch.no_grad():
                    crop_output = model(crop_tensor)
                    pred_mask = torch.sigmoid(crop_output).squeeze().cpu().numpy()
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                full_mask[
                    y : y + window_size,
                    (img.shape[1] // window_size) * window_size : img.shape[1],
                ] += pred_mask[
                    0:224,
                    0 : (img.shape[1] - (img.shape[1] // window_size) * window_size),
                ]

        for x in range(0, img.shape[1] - window_size + 1, window_size):
            crop = img_rgb[
                (img.shape[0] // window_size) * window_size : img.shape[0],
                x : x + window_size,
            ]
            helper = np.zeros((224, 224, 3), dtype=np.uint8)
            helper[
                0 : (img.shape[0] - (img.shape[0] // window_size) * window_size),
                0:224,
                :,
            ] += crop
            crop_tensor = transform(helper).unsqueeze(0).cuda()
            with torch.no_grad():
                crop_output = model(crop_tensor)
                pred_mask = torch.sigmoid(crop_output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            full_mask[
                (img.shape[0] // window_size) * window_size : img.shape[0],
                x : x + window_size,
            ] += pred_mask[
                0 : (img.shape[0] - (img.shape[0] // window_size) * window_size), 0:224
            ]

        if img.shape[0] % window_size != 0:
            crop = img_rgb[
                (img.shape[0] // window_size) * window_size : img.shape[0],
                (img.shape[1] // window_size) * window_size : img.shape[1],
            ]
            helper = np.zeros((224, 224, 3), dtype=np.uint8)
            helper[
                0 : (img.shape[0] - (img.shape[0] // window_size) * window_size),
                0 : (img.shape[1] - (img.shape[1] // window_size) * window_size),
                :,
            ] += crop
            crop_tensor = transform(helper).unsqueeze(0).cuda()
            with torch.no_grad():
                crop_output = model(crop_tensor)
                pred_mask = torch.sigmoid(crop_output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            full_mask[
                (img.shape[0] // window_size) * window_size : img.shape[0],
                (img.shape[1] // window_size) * window_size : img.shape[1],
            ] += pred_mask[
                0 : (img.shape[0] - (img.shape[0] // window_size) * window_size),
                0 : (img.shape[1] - (img.shape[1] // window_size) * window_size),
            ]

    yellow_mask = np.zeros_like(img)
    yellow_mask[full_mask == 255] = [0, 255, 255]
    img = cv2.imread(picture_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(img, 0.7, yellow_mask, 0.5, 0)

    if write:
        cv2.imwrite(save_path, result)

    return result, full_mask
